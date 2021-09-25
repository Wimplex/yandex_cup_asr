import os
import random
import pandas as pd

import torch
from torch.utils.data import DataLoader

from train import predict_classifier
from data import AudioDataset
from networks import ResNet18_Extractor, AudioClassifier
from config import Config


def make_submission(model, test_loader, device, tsv_save_path):
    model = model.to(device)
    model.eval()
    pred, _, names = predict_classifier(model, test_loader, device)
    with open(tsv_save_path, 'w', encoding='utf-8') as file:
        for wav_name, prediction in zip(names, pred):
            w_name = os.path.basename(wav_name)
            file.write('{}\t{}\n'.format(wav_name, prediction))


def submit():
    # Load data
    desc_df = pd.read_csv('desc/desc_train.csv')
    idxs = list(range(len(desc_df)))
    random.shuffle(idxs)
    print(idxs[:40])
    desc_df = desc_df.iloc[idxs[:40], :]
    test_dataset = AudioDataset(desc_df, apply_aug=False)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=2, pin_memory=True)

    # Instantiate model and load weights
    net_kwargs = {
        'input_shape':        Config.INPUT_SHAPE, 
        'num_classes':        Config.NUM_CLASSES,
        'embed_size':         Config.EMB_SIZE,
        'feats_type':         Config.FEATURE_TYPE,
        'feats_n_components': Config.N_COMP,
        'apply_db':           False
    }
    model = ResNet18_Extractor(**net_kwargs)
    model = AudioClassifier(model)
    model.load_state_dict(torch.load('models/classifier_acc0.93712.pth', map_location='cpu'))

    # Make submission
    make_submission(model, test_loader, 'cuda:0', 'submissions/submission.tsv')


if __name__ == '__main__':
    submit()