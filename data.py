import os
import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch.nn as nn
from torch.utils.data import Dataset
from torch_audiomentations import *

from base_utils import load_audio
from config import Config


def create_desc_file_train(train_data_dir, save_path):
    """ Traverses train subdir and creates desc.csv with <wav_name>-<label> pairs """
    label_dirs = glob.glob(os.path.join(train_data_dir, '*'))
    res_df_data = []
    for d in label_dirs:
        res_df_data += [{'wav': d + '/' + wav_name, 'label': d.split('/')[-1]} for wav_name in os.listdir(d)]
    pd.DataFrame(res_df_data).to_csv(save_path, index=False)


def create_desc_file_test(test_data_dir, save_path):
    """ Does the same, but for unlabeled test part """
    wav_paths = glob.glob(os.path.join(test_data_dir, '*.wav'))
    res_df_data = [{'wav': path, 'label': 0} for path in wav_paths]
    pd.DataFrame(res_df_data).to_csv(save_path, index=False)


def tile_wav(wav, max_len=Config.SAMPLE_RATE * 4):
    """ Tiles wav to the right """
    num_reps = int(np.ceil(max_len / wav.shape[-1]))
    wav = wav.repeat([1, num_reps])[:,:max_len]
    return wav


class AudioDataset(Dataset):
    def __init__(self, desc_df, apply_aug=True, noises_dir='', ir_dir=''):
        """
        Represents dataset-wrapper class, organizing wavs loading logic
        :param str data_dir:        dataset directory
        :param str desc_file_path:  .csv-dataframe containing pairs of wav-names and phrases transcription
        :param bool apply_aug:      flag indicating whether augmenttion is needed
        :param str noises_dir:      path to directory with noise .wav-files (only with apply_aug=True)
        :param str ir_dir:          path to directory with impulse response .wav-files (only with apply_aug=True)
        """
        super(AudioDataset, self).__init__()
        self.desc_df = desc_df
        self.noises_dir = noises_dir
        self.ir_dir = ir_dir
        self.target_sr = Config.SAMPLE_RATE
        self.label2idx = {cl: i for i, cl in enumerate(Config.CLASSES_MAP)}
        self.apply_aug = apply_aug
        self.augmentations = Compose([
            ApplyImpulseResponse(ir_dir, p=0.2, sample_rate=Config.SAMPLE_RATE) if ir_dir != '' else nn.Identity(),
            Gain(p=0.3, sample_rate=Config.SAMPLE_RATE),
            AddBackgroundNoise(noises_dir, p=0.9, sample_rate=Config.SAMPLE_RATE, min_snr_in_db=-9, max_snr_in_db=2) if noises_dir != '' else nn.Identity(),
        ], shuffle=True)

    def split_dataset(self, train_size):
        train_df, test_df = train_test_split(self.desc_df, train_size=train_size, stratify=self.desc_df['label'])
        train_dataset = AudioDataset(train_df, apply_aug=True, noises_dir=self.noises_dir, ir_dir=self.ir_dir)
        test_dataset = AudioDataset(test_df, apply_aug=True, noises_dir=self.noises_dir, ir_dir=self.ir_dir)
        return train_dataset, test_dataset

    def __getitem__(self, idx):
        curr_wav_data = self.desc_df.iloc[idx, :]
        wav = load_audio(curr_wav_data['wav'])
        if self.apply_aug:
            wav = self.augmentations(wav.unsqueeze(0)).squeeze(0)
        wav = tile_wav(wav)
        label = curr_wav_data['label']
        label = self.label2idx[label] if label != 0 else label
        return wav, label, curr_wav_data['wav']

    def __len__(self):
        return self.desc_df.shape[0]


if __name__ == '__main__':
    create_desc_file_test(Config.DATA_DIR + 'speech_commands_test', save_path='desc_test.csv')
