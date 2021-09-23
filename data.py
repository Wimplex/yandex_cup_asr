import os
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio.transforms as T
from torch_audiomentations import *
from utils.base_utils import load_audio
from config import Config


def tile_seq_both_sides(signal, max_len=Config.SAMPLE_RATE * 4):
    """ Pad sequence on left and right sides """
    pad_size = max_len - signal.shape[-1]
    if pad_size > 0: 
        signal = nn.ReflectionPad1d([pad_size // 2, pad_size // 2 + 1])(signal.unsqueeze(1)).squeeze(1)
    return signal[:,:max_len]
    

def create_desc_file(train_data_dir, save_path):
    label_dirs = glob.glob(os.path.join(train_data_dir, '*'))
    res_df_data = []
    for d in label_dirs:
        res_df_data += [{'wav': d + '/' + wav_name, 'label': d.split('/')[-1]} for wav_name in os.listdir(d)]
    pd.DataFrame(res_df_data).to_csv(save_path, index=False)


class AudioDataset(Dataset):
    def __init__(self, desc_df, label2idx=None):
        """
        Represents dataset-wrapper class, organizing wavs loading logic
        :param str data_dir:        dataset directory
        :param str desc_file_path:  .csv-dataframe containing pairs of wav-names and phrases transcription
        """
        super(AudioDataset, self).__init__()
        self.desc_df = desc_df
        self.target_sr = Config.SAMPLE_RATE
        self.label2idx = label2idx if label2idx is not None else self.__create_label2idx_mapping()    
        
    def __create_label2idx_mapping(self):
        idx2label = self.desc_df['label'].unique()
        return {label: idx for idx, label in enumerate(idx2label)}

    def __getitem__(self, idx):
        curr_wav_data = self.desc_df.iloc[idx, :]
        wav = load_audio(curr_wav_data['wav'], self.target_sr)
        wav = tile_seq_both_sides(wav)
        label = self.label2idx[curr_wav_data['label']]
        return wav, label

    def __len__(self):
        return self.desc_df.shape[0]


class Preprocessor(nn.Module):
    def __init__(self, feats_type, n_components, noises_dir, ir_dir, to_db=False):
        super(Preprocessor, self).__init__()
        """
        Implements data augmentation and feature extraction logic
        :param str feats_type ['mels' | 'mfcc']:    features type
        :param str noises_dir:                      path to directory with noise .wav-files
        :param str ir_dir:                          path to directory with impulse response .wav-files
        :param bool to_db:                          organizes amplitude-to-db conversion
        """
        # This augmentations order has chosen as the best variant of producing natural sounds.
        # So, the 'shuffle' argument could be False.
        self.augmentations = Compose([
            ApplyImpulseResponse(ir_dir, p=0.3, sample_rate=Config.SAMPLE_RATE),
            Gain(p=0.4, sample_rate=Config.SAMPLE_RATE),
            AddBackgroundNoise(noises_dir, p=0.4, sample_rate=Config.SAMPLE_RATE),
            AddColoredNoise(min_f_decay=0, max_f_decay=0.1, sample_rate=Config.SAMPLE_RATE)
        ], shuffle=True)

        if feats_type == 'mels':
            self.features_ext = Compose([
                T.MelSpectrogram(
                    Config.SAMPLE_RATE, n_fft=Config.N_FFT, hop_length=Config.HOP_LEN, n_mels=n_components, normalized=True),
                T.AmplitudeToDB() if to_db else nn.Identity()
            ])
        elif feats_type == 'mfcc':
            self.features_ext = Compose([
                T.MFCC(Config.SAMPLE_RATE, n_mfcc=n_components, melkwargs={'n_fft': Config.N_FFT, 'hop_length': Config.HOP_LEN}),
                T.SlidingWindowCmn(cmn_window=300, norm_vars=True),
                T.AmplitudeToDB() if to_db else nn.Identity()
            ])
        self.delta_ext = T.ComputeDeltas(win_length=5)

    def forward(self, x):
        x = self.augmentations(x)
        feats = self.features_ext(x)
        delta = self.delta_ext(feats)
        delta2 = self.delta_ext(delta)
        out = torch.cat([feats, delta, delta2], axis=1)
        return out


if __name__ == '__main__':
    dset = AudioDataset('desc.csv')
    entity, label = next(iter(dset))
    entity = entity.unsqueeze(1)

    preproc = Preprocessor('mels', 64, noises_dir='data/noises', ir_dir='data/ir', to_db=True)
    print(entity.shape)
    print(preproc(entity).shape)
