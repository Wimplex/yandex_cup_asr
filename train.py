import warnings 
warnings.filterwarnings('ignore')

import tqdm
import random

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from networks import *
from data import AudioDataset
from base_utils import save_model


def evaluate_loss(model, test_loader, device, criterion='model'):
    """ Returns average loss on test set for input model """
    losses = []
    for batch_X, batch_y, _ in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        if criterion == 'model': loss = model.forward_loss(batch_X, batch_y)
        else:
            output_batch = model(batch_X)
            loss = criterion(output_batch, batch_y)
        losses.append(loss.item())
    return np.mean(losses)
        

def predict_classifier(model, data_loader, device):
    """ Returns predictions from classifier model """
    is_model_in_train_mode = model.training
    model = model.to(device)
    if is_model_in_train_mode: model.eval()
    with torch.no_grad():
        pred, true = torch.Tensor().to(device), torch.Tensor().to(device)
        names = []
        for batch_X, batch_y, batch_name in tqdm.tqdm(data_loader, total=len(data_loader)):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_out = model(batch_X)
            pred = torch.cat([pred, batch_out.argmax(axis=1)])
            true = torch.cat([true, batch_y])
            names += batch_name
    if is_model_in_train_mode: model.train()
    return pred.cpu().int(), true.cpu(), names


def train_extractor(model, train_loader, test_loader, \
                    optimizer, scheduler, n_epochs, eval_every):
    device = model.device
    model.train()
    best_loss = 15.0
    last_train_loss, last_eval_loss = 0, 0

    for ep in range(n_epochs):
        print("Epoch %s/%s started." % (ep + 1, n_epochs))

        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        train_loss = []
        for i, (batch_X, batch_y, _) in pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model.forward_loss(batch_X, batch_y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            curr_iter = ep * len(train_loader) + i
            if curr_iter % 10 == 0:
                last_train_loss = np.mean(train_loss)
                
            if curr_iter % eval_every == 0:
                last_eval_loss = evaluate_loss(model, test_loader, device)
                if best_loss > last_eval_loss:
                    best_loss = last_eval_loss
                    print('New best loss: %s!' % np.round(best_loss, 4))
                    save_model(model, 'models/extractor.pth')
                scheduler.step(last_eval_loss)
            
            pbar.set_description("train_loss: %s, eval_loss: %s" % (last_train_loss, last_eval_loss))
            torch.cuda.empty_cache()


def train_classifier(model, train_loader, test_loader, \
                     optimizer, scheduler, n_epochs, eval_every):
    device = model.device
    model.train()
    last_train_loss, best_acc = 0, 0.0

    for ep in range(n_epochs):
        print("Epoch %s/%s started." % (ep + 1, n_epochs))

        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        metrics = {'train_loss': [], 'eval_loss': []}
        for i, (batch_X, batch_y, _) in pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output_batch = model(batch_X)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(output_batch, batch_y)
            loss.backward()
            optimizer.step()
            metrics['train_loss'].append(loss.item())

            curr_iter = ep * len(train_loader) + i
            if curr_iter % 10 == 0:
                last_train_loss = np.mean(metrics['train_loss'])
                pbar.set_description("train_loss: %s, best_eval_accuracy: %s" % (last_train_loss, best_acc))

            if curr_iter % eval_every == 0:
                pred, true, _ = predict_classifier(model, test_loader, device)
                acc = accuracy_score(true, pred)

                if acc > best_acc:
                    print('New best accuracy: %s!' % acc)
                    best_acc = acc
                    save_model(model, 'models/classifier.pth')
                
                scheduler.step(acc)
            torch.cuda.empty_cache()


def run(train_type, train_loader, test_loader):
    # Training preparations
    device = Config.DEVICE
    print("Training on", device)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    # Setup randomness
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    torch.cuda.manual_seed(Config.SEED)
    torch.backends.cudnn.deterministic = True

    # Instantiate model
    net_kwargs = {
        'input_shape':        Config.NUM_CHANNELS,
        'num_classes':        Config.NUM_CLASSES,
        'embed_size':         Config.EMB_SIZE,
        'feats_type':         Config.FEATURE_TYPE,
        'feats_n_components': Config.N_COMP,
        'apply_db':           False
    }
    model = ResNet18_Extractor(**net_kwargs)
    # model = EfficientNet_Extractor(effnet_type=Config.MODEL.split('-')[-1], **net_kwargs)
    if train_type == 'classifier':
        model.load_state_dict(torch.load('models/extractor.pth'))
        model = AudioClassifier(model)
    model = model.to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, betas=(1e-3, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.85, mode='min')

    if train_type == 'extractor':
        train_extractor(
            model,
            train_loader=train_loader, 
            test_loader=test_loader, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            n_epochs=Config.NUM_EPOCHS_EXTRACTOR, 
            eval_every=100
        )
    elif train_type == 'classifier':
        train_classifier(
            model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=Config.NUM_EPOCHS_CLASSIFIER,
            eval_every=100
        )


if __name__ == '__main__':
    # Create dataloaders
    desc_df = pd.read_csv('desc/desc_train.csv')
    train_dataset, test_dataset = AudioDataset(desc_df, noises_dir=Config.DATA_DIR + 'noises', 
                                            ir_dir='data/ir').split_dataset(train_size=0.97)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    print("Train size (num of tensors):", len(train_dataset))
    print("Test size (num of tensors):", len(test_dataset))

    # Run extractor training
    run('extractor', train_loader, test_loader)

    # Run classifier training
    run('classifier', train_loader, test_loader)

# TODO "Что не забыть на соревновании":
#   - Поиграться с аргументом normalize у torchaudio.load() (лежит в utils.py)
#   - Посчитать максимальную длительность речи на незашумленных данных