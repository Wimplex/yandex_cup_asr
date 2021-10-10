from re import sub
import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd

import torch
from torch.utils.data import DataLoader

from config import Config
from utils.inference_utils import predict_classifier
from utils.data_utils import AudioDataset
from modules.networks import ResNet18_Extractor, AudioClassifier


def make_submission(model, test_loader, device, tsv_save_path):
    """ Creates 'submission.tsv' file """
    pred, _, names = predict_classifier(model, test_loader, device)
    with open(tsv_save_path, 'w', encoding='utf-8') as file:
        for wav_name, prediction in zip(names, pred):
            w_name = os.path.basename(wav_name)
            file.write('{}\t{}\n'.format(w_name, Config.CLASSES_MAP[prediction]))


def repair_submission(submission_file_path):
    df = pd.read_csv(submission_file_path, sep='\t')
    df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: Config.CLASSES_MAP[x])
    df.to_csv(os.path.join(os.path.dirname(submission_file_path), 'submission_repaired.tsv'), index=False, sep='\t')



def submit():
    # Load data
    desc_df = pd.read_csv('desc/desc_test.csv')
    test_dataset = AudioDataset(
        desc_df,
        classes_list=Config.CLASSES_MAP,
        sample_rate=Config.SAMPLE_RATE
    )
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=2, pin_memory=True)

    # Instantiate model and load weights
    net_kwargs = {
        'input_shape': [Config.NUM_CHANNELS, 300, 300],
        'num_classes': Config.NUM_CLASSES,
        'embed_size':  Config.EMB_SIZE,
    }
    model = ResNet18_Extractor(**net_kwargs)
    model = AudioClassifier(model)

    st_dict = torch.load('models/classifier.pth')
    model.load_state_dict({key: val for key, val in st_dict.items() if key.split('.')[1] != 'preproc'})
    # model.load_state_dict(torch.load('models/classifier_acc0.93712.pth', map_location='cpu'))

    # Make submission
    make_submission(model, test_loader, 'cuda:0', 'submissions/submission.tsv')


if __name__ == '__main__':
    submit()
