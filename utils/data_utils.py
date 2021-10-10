import os
import glob
import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
import soundfile

import torch
from torch.utils.data import Dataset
from torch_audiomentations import *

# from modules.unet_vad import UNetVad


def create_desc_file_train(train_data_dir, save_path):
    """ Traverses train subdir and creates desc.csv with <wav_name>-<label> pairs """
    label_dirs = glob.glob(os.path.join(train_data_dir, '*.npy'))
    res_df_data = []
    for d in label_dirs:
        res_df_data += [{'wav': d + '/' + wav_name, 'label': d.split('/')[-1]} for wav_name in os.listdir(d)]
    pd.DataFrame(res_df_data).to_csv(save_path, index=False)


def create_desc_file_test(test_data_dir, save_path):
    """ Does the same, but for unlabeled test part """
    wav_paths = glob.glob(os.path.join(test_data_dir, '*.npy'))
    res_df_data = [{'wav': path, 'label': 0} for path in wav_paths]
    pd.DataFrame(res_df_data).to_csv(save_path, index=False)


def tile_seq(seq, max_len):
    """ Tiles wav to the right (UNUSED, perhaps) """
    num_reps = int(np.ceil(max_len / seq.shape[-1]))
    seq = seq.repeat([1, num_reps])[:,:max_len]
    return seq


def random_crop(x, crop_width):
    """ Random crop over time axis """
    start = np.random.randint(0, x.shape[1] - crop_width)
    return x[:, start:start+crop_width, :]


def add_background_noise(wav, noise, noise_ampl):
    """ Applies background noise onto input wav """
    # Trim noise
    num_reps = wav.shape[0] // noise.shape[0] + 2
    noise = noise.repeat(num_reps, axis=0)
    start_idx = np.random.randint(0, noise.shape[0] - wav.shape[0] + 1)
    noise = noise[start_idx:start_idx + wav.shape[0]]

    # Apply noise
    noise_mult = np.abs(wav).max() / np.abs(noise).max() * noise_ampl
    return (wav + noise_mult * noise) / (1 + noise_ampl)


def extract_features(wav, sr, features_type, n_components, win_len, hop_len, normalize=False):
    """ Extracts features (mfcc/mels, d, dd) from waveform with specific params """
    if features_type == 'mfcc':
        features = librosa.feature.mfcc(
            wav, sr, n_mfcc=n_components, n_fft=win_len, hop_length=hop_len).T
    elif features_type == 'mels':
        features = librosa.feature.melspectrogram(
            wav, n_mels=n_components, n_fft=win_len, hop_length=hop_len).T
    if normalize: features = (features - features.mean()) / features.std()

    d1 = librosa.feature.delta(features, axis=0)
    d2 = librosa.feature.delta(features, order=2, axis=0) 
    features = np.stack([features, d1, d2], axis=0)
    return features


def resample_dir(audio_dir, target_sr):
    """ Resamples all .wav-files in a directory to target_sr """
    wav_paths = glob.glob(os.path.join(audio_dir, '*.wav'))
    for wav_p in tqdm.tqdm(wav_paths):
        wav, sr = librosa.load(wav_p, None)
        if sr != target_sr: wav = librosa.resample(wav, sr, target_sr)
        soundfile.write(wav_p, wav, target_sr)


class AudioDataset(Dataset):
    def __init__(self, desc_df, classes_list, frame_size):
        """ 
            Dataset-wrapper for audio data
            :param pd.DataFrame desc_df: Dataframe object, containing wavs paths and corresponding labels
            :param list classes_list:    List of labels available in dataset
            :param int frame_size:       Number of time-domain samples cropping from data object
        """
        self.desc_df = desc_df
        self.classes_list = classes_list
        self.class2idx = {val: i for i, val in enumerate(classes_list)}
        self.frame_size = frame_size

    def split_dataset(self, train_size):
        train_df, test_df = train_test_split(
            self.desc_df, 
            train_size=train_size, 
            stratify=self.desc_df['label']
        )
        return AudioDataset(train_df, self.classes_list, self.frame_size), \
               AudioDataset(test_df, self.classes_list, self.frame_size)

    def __getitem__(self, idx):
        curr_item = self.desc_df.iloc[idx,:]
        X = np.load(curr_item['wav'])
        vad_markup = np.load(curr_item['vad'])
        label = curr_item['label']

        # Crop or fill with zeros input features
        if X.shape[1] < self.frame_size:
            # Pad X
            dummy_X = np.zeros([X.shape[0], self.frame_size, X.shape[2]])
            start_idx = self.frame_size
            dummy_X[:,start_idx:start_idx + X.shape[1]] = X
            X = dummy_X
            # Pad vad markup
            vad_dummy = np.zeros([self.frame_size])
            vad_dummy[start_idx:start_idx + X.shape[1]] = vad_markup
            vad_markup = vad_dummy
        else:
            # Get most reliable windows according to VAD markup. Preserves probs.
            if vad_markup.shape[0] > X.shape[1]:
                dummy_X = np.zeros([X.shape[0], vad_markup.shape[0], X.shape[2]])
                dummy_X[:,:X.shape[1]] = X
                X = dummy_X
            speech_ids = np.argpartition(vad_markup, -self.frame_size)[-self.frame_size:] # <-- maximum values without sorting with np.argsort
            X = X[:,speech_ids]
            vad_markup = vad_markup[speech_ids]

        # Process label
        label = self.class2idx[label] if label != 0 else 0
        return torch.Tensor(X), label, torch.Tensor(vad_markup), curr_item['wav']

    def __len__(self):
        return len(self.desc_df)


class PredictionAudioDataset(Dataset):
    def __init__(self, desc_df, classes_list, frame_size, num_frames):
        """ 
            Dataset-wrapper for audio data
            :param pd.DataFrame desc_df: Dataframe object, containing wavs paths and corresponding labels
            :param list classes_list:    List of labels available in dataset
            :param int frame_size:       Number of time-domain samples cropping from data object
            :param int num_frames:       Number of frames cropped from data object (for more robust predictions)
        """
        self.desc_df = desc_df
        self.classes_list = classes_list
        self.class2idx = {val: i for i, val in enumerate(classes_list)}
        self.frame_size = frame_size
        self.num_frames = num_frames

    def __getitem__(self, idx):
        # Get current object
        obj_idx = idx // self.num_frames
        curr_item = self.desc_df.iloc[obj_idx,:]
        X = np.load(curr_item['wav'])

        # Crop X frame-wise according to current input idx
        curr_frame_idx = idx % self.num_frames
        frame_step = int((X.shape[0] - self.frame_size) / (self.num_frames - 1))
        curr_frame_start = frame_step * curr_frame_idx
        X = X[curr_frame_start:curr_frame_start + self.frame_size]

        # Get label
        label = curr_item['label']
        label = self.class2idx[label] if label != 0 else 0
        return X, label, curr_item['wav']

    def __len__(self):
        return len(self.desc_df) * self.num_frames