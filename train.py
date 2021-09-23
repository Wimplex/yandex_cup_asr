import os
import tqdm
import random
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from networks import *
from data import AudioDataset, Preprocessor
from utils.base_utils import save_model


def evaluate_loss(model, test_loader, preprocessor, device, criterion='model'):
    """ Returns average loss on test set for input model """

    losses = []
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        batch_X = preprocessor(batch_X)
        if criterion == 'model': loss = model(batch_X, batch_y)
        else:
            output_batch = model(batch_X)
            loss = criterion(output_batch, batch_y)
        losses.append(loss.item())
    return np.mean(losses)
        

def predict_classifier(model, preprocessor, data_loader, device):
    """ Returns predictions from classifier model """

    is_model_in_train_mode = model.training
    model = model.to(device)
    if is_model_in_train_mode: model.eval()
    with torch.no_grad():
        pred, true = torch.Tensor().to(device), torch.Tensor().to(device)
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_X = preprocessor(batch_X)
            batch_out = model(batch_X)
            pred = torch.cat([pred, batch_out.argmax(axis=1)])
            true = torch.cat([true, batch_y])
    if is_model_in_train_mode: model.train()
    return pred.cpu(), true.cpu()


def train_extractor(model, preprocessor, train_loader, test_loader, optimizer, scheduler, n_epochs, device, eval_every):
    preprocessor = preprocessor.to(device)
    model = model.to(device)
    model.train()
    best_loss = 10.0

    for ep in range(n_epochs):
        print("Epoch %s/%s started." % (ep + 1, n_epochs))

        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        metrics = {'train_loss': [], 'eval_loss': []}
        for i, (batch_X, batch_y) in pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_X = preprocessor(batch_X)
            optimizer.zero_grad()
            loss = model(batch_X, batch_y)
            loss.backward()
            optimizer.step()
            metrics['train_loss'].append(loss.item())

            curr_iter = ep * len(train_loader) + i
            if curr_iter % 10 == 0:
                pbar.set_description("train_loss: %s" % np.mean(metrics['train_loss']))

            if curr_iter % eval_every:
                eval_loss = evaluate_loss(model, test_loader, preprocessor, device)
                suffix = pbar.suffix
                pbar.set_description(suffix + ', eval_loss: %s' % eval_loss)

                if eval_loss < best_loss:
                    print('New best loss')
                    save_model(model, 'models/ep%s_iter%s_loss%s.pth' % (ep, curr_iter, np.round(eval_loss, 4)))
        scheduler.step(eval_loss)


def train_classifier(model, train_loader, test_loader, optimizer, scheduler, n_epochs, device, eval_every):
    model.to(device)
    model.train()
    pass



def main():
    # Setup randomness
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    torch.cuda.manual_seed(Config.SEED)
    torch.backends.cudnn.deterministic = True

    # Training preparations
    device = 'cuda:0'
    torch.cuda.empty_cache()

    # Instantiate model
    net_kwargs = {'input_shape': Config.INPUT_SHAPE, 'num_classes': Config.NUM_CLASSES, 'embed_size': Config.EMB_SIZE}
    if Config.MODEL.startswith('efficientnet-'):
        model = EfficientNet_Extractor(effnet_type=Config.MODEL.split('-')[-1], **net_kwargs)
    else:
        if Config.MODEL == 'deit': net_cls = DeiT_Extractor
        elif Config.MODEL == 'resnet18': net_cls = ResNet18_Extractor
        model = net_cls(**net_kwargs)

    # Split data
    desc_df = pd.read_csv('desc.csv')
    train_df, test_df = train_test_split(desc_df, test_size=0.03, stratify=desc_df['label'])
    print("Train size:", len(train_df))
    print("Test size:", len(test_df))

    # Create dataloaders
    train_dataset = AudioDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_dataset = AudioDataset(test_df, label2idx=train_dataset.label2idx)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # Create preprocessing unit
    preproc = Preprocessor(
        feats_type=Config.FEATURE_TYPE, 
        n_components=Config.N_COMP, 
        noises_dir='data/noises', 
        ir_dir='data/ir'
    )

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, betas=(1e-4, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.75, mode='min')

    train_extractor(
        model, 
        preprocessor=preproc, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        n_epochs=Config.NUM_EPOCHS, 
        device=device,
        eval_every=100
    )


if __name__ == '__main__':
    main()


# TODO "Что не забыть на соревновании":
#   - Понять, как организованы описательные данные (например, description.csv) в предоставленном датасете
#   - Поиграться с аргументом normalize у torchaudio.load() (лежит в utils.py)
#   - Проверить количество всех лейблов в данных
#   - Применить one-hot-кодирование к лейблам.
#   - Посчитать максимальную длительность речи на незашумленных данных