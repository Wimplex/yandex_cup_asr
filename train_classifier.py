import warnings
warnings.filterwarnings('ignore')

import os
import tqdm
import random

import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from modules.extractors import *
from utils.base_utils import save_model, save_checkpoint, now
from utils.data_utils import AudioDataset
from utils.inference_utils import evaluate_loss, predict_classifier
from utils.metrics import accuracy


def train_extractor(model, train_loader, test_loader, \
                    optimizer, scheduler, n_epochs, eval_every):
    """ Runs extractor model training process """
    device = model.device
    model.train()
    best_loss = 15.0
    last_train_loss, last_eval_loss = 0, 0

    for ep in range(n_epochs):
        print("Epoch %s/%s started." % (ep + 1, n_epochs))

        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        train_loss = []
        for i, (batch_X, batch_y, batch_vad, _) in pbar:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_vad = batch_vad.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model.forward(batch_X, batch_vad, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())

            # Save checkpoint
            curr_iter = ep * len(train_loader) + i
            if curr_iter != 0 and ep > 0 and curr_iter % 700 != 0:
                print("Saving checkpoint")
                fname = f'extractor_{now()}_lastbestloss_{np.round(best_loss, 3)}.ckpt'
                save_checkpoint(model, optimizer, f'models/extractor/checkpoints/{fname}')

            # Compute mean train loss
            if curr_iter % 10 == 0: last_train_loss = np.mean(train_loss)

            # Compute mean eval loss
            #if curr_iter != 0 and curr_iter % eval_every == 0:
            if curr_iter % eval_every == 0:
                last_eval_loss = evaluate_loss(model, test_loader, device)
                if best_loss > last_eval_loss:
                    best_loss = last_eval_loss
                    print('New best eval loss: %s!' % np.round(best_loss, 4))
                    save_model(model, 'models/extractor.pth')
            pbar.set_description("train_loss: {}, eval_loss: {}, lr: {}".format(
                np.round(last_train_loss, 4), 
                np.round(last_eval_loss, 4), 
                np.round(scheduler.get_last_lr()[0], 6)
            ))
            torch.cuda.empty_cache()


def train_classifier(model, train_loader, test_loader, \
                     optimizer, scheduler, n_epochs, eval_every):
    """ Runs classifier model training process """
    device = model.device
    model.train()
    last_train_loss, best_acc, last_acc = 0, 0.0, 0.0

    for ep in range(n_epochs):
        print("Epoch %s/%s started." % (ep + 1, n_epochs))

        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        train_loss = []
        for i, (batch_X, batch_y, _) in pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output_batch = model(batch_X)
            
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(output_batch, batch_y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            # Save checkpoint
            curr_iter = ep * len(train_loader) + i
            if curr_iter != 0 and ep > 0 and curr_iter % 500 != 0:
                print("Saving checkpoint")
                fname = f'classifier_{now()}_lastbestacc_{np.round(best_acc, 3)}.ckpt'
                save_checkpoint(model, optimizer, f'models/extractor/checkpoints/{fname}')

            # Calculate mean train loss
            if curr_iter % 10 == 0: last_train_loss = np.mean(train_loss)

            # Evaluate test metrics
            if curr_iter % eval_every == 0:
                pred, true, _ = predict_classifier(model, test_loader, device)
                last_acc = accuracy(pred, true)
                if last_acc > best_acc:
                    print('New best accuracy: %s!' % last_acc)
                    best_acc = last_acc
                    save_model(model, 'models/classifier.pth')
                scheduler.step(last_acc)

            pbar.set_description("train_loss: %s, eval_accuracy: %s" % (last_train_loss, last_acc))
            torch.cuda.empty_cache()


def run(train_type, train_loader, test_loader):
    """ Runs training """
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
        'input_shape': Config.NUM_CHANNELS,
        'num_classes': Config.NUM_CLASSES,
        'embed_size':  Config.EMB_SIZE,
    }
    model = ResNet18_Extractor(**net_kwargs)
    if train_type == 'classifier':
        model = AudioClassifier(model)
    model = model.to(device)

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.05, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, betas=(0.86, 0.999))

    try:
        if train_type == 'extractor':      
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr          = 0.01, 
                epochs          = Config.NUM_EPOCHS_EXTRACTOR, 
                steps_per_epoch = len(train_loader)
            )
            train_extractor(
                model,
                train_loader = train_loader, 
                test_loader  = test_loader, 
                optimizer    = optimizer, 
                scheduler    = scheduler, 
                n_epochs     = Config.NUM_EPOCHS_EXTRACTOR, 
                eval_every   = Config.EVAL_EVERY
            )
        elif train_type == 'classifier':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=0.75, mode='max')
            train_classifier(
                model,
                train_loader = train_loader,
                test_loader  = test_loader,
                optimizer    = optimizer,
                scheduler    = scheduler,
                n_epochs     = Config.NUM_EPOCHS_CLASSIFIER,
                eval_every   = Config.EVAL_EVERY
            )
    except KeyboardInterrupt:
        print("Training process ended manually. Saving checkpoint.")
        save_checkpoint(
            model, optimizer, f'models/extractor/checkpoints/{train_type}_{now()}.ckpt')
        exit()


if __name__ == '__main__':
    # Create and split dataset 
    desc_df = pd.read_csv(os.path.join(Config.DESC_DIR, 'desc_train_mfcc_400_160.csv'))
    dataset = AudioDataset(
        desc_df,
        Config.CLASSES_MAP,
        frame_size=Config.FRAME_SIZE
    )
    train_dataset, test_dataset = dataset.split_dataset(train_size=0.985)
    print("Train size (num of tensors):", len(train_dataset))
    print("Test size (num of tensors):", len(test_dataset))

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
    
    # Run extractor training
    run('extractor', train_loader, test_loader)

    # Run classifier training
    run('classifier', train_loader, test_loader)
