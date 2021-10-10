import warnings
warnings.filterwarnings('ignore')

import tqdm
import random

import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from modules.vad import UNetVad
from modules.losses import BCESoftJaccardDice
from utils.vad_utils import VadDataset
from utils.base_utils import save_checkpoint, save_model, now
from utils.metrics import eer_batches



def predict(model, data_loader, device):
    """ Predicts speech labels for input data_loader """
    is_model_training = model.training
    if is_model_training: model.eval()
    if str(model.device) != device: model = model.to(device)
    
    pred = torch.Tensor().to(device)
    true = torch.Tensor().to(device)
    with torch.no_grad():
        for (batch_X, batch_y) in tqdm.tqdm(data_loader, total=len(data_loader)):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_out = model(batch_X)
            pred = torch.cat([pred, batch_out])
            true = torch.cat([true, batch_y])

    if is_model_training: model.train()
    return true.detach().cpu().numpy(), pred.detach().cpu().numpy()


def train(model, train_loader, test_loader, optimizer, scheduler, criterion, n_epochs, eval_every):
    model.train()
    device = model.device()

    last_train_loss, last_eer_thr = 0, 0.0
    best_eer = last_eval_eer = 0.5

    pbar = tqdm.tqdm(train_loader, total=len(train_loader))
    for ep in range(n_epochs):
        train_losses = []
        for i, (batch_X, batch_y) in enumerate(pbar):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_out = model(batch_X)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(batch_out, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

            curr_iter = ep * len(train_loader) + i
            if curr_iter % 10 == 0:
                last_train_loss = np.round(np.mean(train_losses), 4)

            # if curr_iter != 0 and curr_iter % eval_every == 0:
            if curr_iter % eval_every == 0:
                true, pred = predict(model, test_loader, device)
                last_eval_eer, last_eer_thr = eer_batches(true, pred)
                if last_eval_eer < best_eer:
                    best_eer = last_eval_eer
                    print("New best eval eer: %s!" % np.round(best_eer, 4))
                    state_dict = {
                        'model': model.state_dict(),
                        'thr': last_eer_thr
                    }
                    save_model(state_dict, 'models/vad.pth')

            pbar.set_description("train_loss: {}, eval_eer: {}".format(
                np.round(last_train_loss, 4), np.round(last_eval_eer, 4)
            ))
            torch.cuda.empty_cache()

def run(train_loader, test_loader):
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
    model = UNetVad(Config.VAD_ARCH_SHAPE)
    model = model.to(device)
    # model.initialize_weights()

    # Instantiate optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE_VAD)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.01, 
        epochs=Config.NUM_EPOCHS_VAD,
        steps_per_epoch=len(train_loader)
    )

    # Instantiate loss
    criterion = BCESoftJaccardDice(mode=Config.VAD_LOSS_TYPE)

    # Run training process
    try:
        train(model, train_loader, test_loader, 
            optimizer, scheduler, criterion, 
            Config.NUM_EPOCHS_VAD, eval_every=Config.EVAL_EVERY)
    except KeyboardInterrupt:
        print("Training process ended manually. Saving checkpoint.")
        save_checkpoint(model, optimizer, f'models/vad_checkpoints/checkpoint_{now()}.ckpt')
        exit()



if __name__ == '__main__':
    feats2chunks_df = pd.read_csv('desc/desc_features2chunks.csv')
    train_dataset, test_dataset = VadDataset(feats2chunks_df, mode='segment').split(train_size=0.9998)
    print("Train size: %s, Test size: %s" % (len(train_dataset), len(test_dataset)))
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
    run(train_loader, test_loader)
