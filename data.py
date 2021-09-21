import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio.transforms as T
from torch_audiomentations import *
from utils.base_utils import load_audio
from config import Config


class AudioDataset(Dataset):
    def __init__(self, data_dir, desc_file_path, target_sr):
        """
        Represents dataset-wrapper class, organizing wavs loading logic
        :param str data_dir:        dataset directory
        :param str desc_file_path:  .csv-dataframe containing pairs of wav-names and phrases transcription
        :param int target_sr:       Target sample rate to resample to
        """
        super(AudioDataset, self).__init__()
        self.data_dir = data_dir
        self.desc_df = pd.read_csv(desc_file_path)
        self.target_sr = target_sr

    def __getitem__(self, idx):
        curr_wav_path = os.path.join(self.data_dir, self.desc_df.iloc[idx, :]['name'].values[0])
        wav = load_audio(curr_wav_path, self.target_sr)
        label = None
        return wav, label

    def __len__(self):
        return self.desc_df.shape[0]


class Preprocessor(nn.Module):
    def __init__(self, feats_type, noises_dir, ir_dir, to_db=False):
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
            AddBackgroundNoise(noises_dir, p=0.3, sample_rate=Config.SAMPLE_RATE),
            AddColoredNoise(min_f_decay=0, max_f_decay=0.1, sample_rate=Config.SAMPLE_RATE)
        ], shuffle=True)

        if feats_type == 'mels':
            self.features_ext = Compose([
                T.MelSpectrogram(
                    Config.SAMPLE_RATE, n_fft=Config.N_FFT, hop_length=Config.HOP_LEN, n_mels=Config.N_COMP, normalized=True),
                T.AmplitudeToDB() if to_db else nn.Identity()
            ])
        elif feats_type == 'mfcc':
            self.features_ext = Compose([
                T.MFCC(Config.SAMPLE_RATE, n_mfcc=Config.N_COMP, melkwargs={'n_fft': Config.N_FFT, 'hop_length': Config.HOP_LEN}),
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
        print(out.shape)
        return out
