import os
import glob
import tqdm
from collections import Counter
import multiprocessing as mp

import librosa
import matplotlib.pyplot as plt
import numpy as np
from config import Config


def get_wav_len(wav_path):
    return librosa.load(wav_path, sr=None)[0].shape[1]


def load_wavs_paths(path):
    wav, sr = librosa.load(path, sr=None)
    return sr, wav.shape[0]


def get_stats_parallel(paths):
    pool = mp.Pool(7)
    res = tqdm.tqdm(pool.imap(load_wavs_paths, paths), total=len(paths))
    pool.close()
    return res


def get_data_sample_rates():
    # Train part
    wavs_paths = glob.glob(os.path.join(Config.DATA_DIR, 'speech_commands_train', '*', '*.wav'))
    train_srs = [librosa.load(p, sr=None)[1] for p in tqdm.tqdm(wavs_paths)]
    
    # Test part
    wavs_paths = glob.glob(os.path.join(Config.DATA_DIR, 'speech_commands_test', '*.wav'))
    test_srs = [librosa.load(p, sr=None)[1] for p in tqdm.tqdm(wavs_paths)]
    return Counter(train_srs), Counter(test_srs)


def wav_length_distribution():
    # Train part
    wavs_paths = glob.glob(os.path.join(Config.DATA_DIR, 'speech_commands_train', '*', '*.wav'))
    lens = [librosa.get_duration(filename=p) for p in tqdm.tqdm(wavs_paths[:10000])]
    plt.hist(lens, bins=200)
    plt.savefig('img/len_dist_train.png', dpi=100)
    plt.close()

    # Test part
    wavs_paths = glob.glob(os.path.join(Config.DATA_DIR, 'speech_commands_test', '*.wav'))
    lens = [librosa.get_duration(filename=p) for p in tqdm.tqdm(wavs_paths[:10000])]
    plt.hist(lens, bins=200)
    plt.savefig('img/len_dist_test.png', dpi=100)
    plt.close()


if __name__ == '__main__':

    # All files are 16rHz signals
    train_srs, test_srs = get_data_sample_rates()
    print("Train sample rates:", train_srs)
    print("Test sample rates:", test_srs)

    # Min_len == 1.3 sec, Max_len == 4.0 sec
    wav_length_distribution()


