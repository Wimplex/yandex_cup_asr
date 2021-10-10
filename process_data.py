import warnings
warnings.filterwarnings('ignore')

import os
import glob
import tqdm

import numpy as np
import pandas as pd
import librosa

import torch

from config import Config
from utils.base_utils import check_dir
from utils.data_utils import add_background_noise, extract_features, \
                             create_desc_file_train, create_desc_file_test
from utils.vad_utils import extract_and_save_reference_seg, predict_full_overlapped, \
                             create_chunks2aug_features_desc_file
from modules.vad import UNetVad


def prepare_project_tree():
    # Prepare data and features dirs
    check_dir(Config.FEATURES_DIR)
    check_dir(Config.VAD_REF_DIR)
    check_dir(Config.VAD_SYS_DIR)
    check_dir(Config.EMBEDDINGS_DIR)

    # Prepare models dir
    check_dir('models')
    check_dir('models/vad')
    check_dir('models/vad/checkpoints')
    check_dir('models/extractor')
    check_dir('models/extractor/checkpoints')

    # Prepare desc, img and submissions dir
    check_dir(Config.DESC_DIR)
    check_dir('img')
    check_dir('submissions')


def apply_augmentations_dir(wavs_list, noises_list, output_dir, \
                            num_augmentations, min_noise_ampl, max_noise_ampl, \
                            features_type, n_components, win_len, hop_len, normalize):
    """ Applies augmentations and extracts features for all wavs in a dir """
    for wav_p in tqdm.tqdm(wavs_list):
        wav_base_name = os.path.basename(wav_p)
        wav, sr = librosa.load(wav_p, sr=None)
        noises_paths = np.random.choice(noises_list, num_augmentations)
        for i, noise_p in enumerate(noises_paths):
            noise_ampl = np.random.uniform(min_noise_ampl, max_noise_ampl)
            noise, _ = librosa.load(noise_p, sr=None) 
            aug_wav = add_background_noise(wav, noise, noise_ampl)
            features = extract_features(aug_wav, sr, features_type, n_components, win_len, hop_len, normalize)

            save_name = '%s_snr_%s_%s.npy' % (wav_base_name.split('.')[0], np.round(noise_ampl, 3), i + 1)
            save_path = os.path.join(output_dir, save_name)
            np.save(save_path, features)


def preprocess_train(train_dir, noises_dir, output_dir, create_desc_csv=True, **kwargs):
    """ Extracts features from all over train directories tree's wavs """
    print("Augmenting and extracting features for Train part of dataset.")
    noises_paths = glob.glob(os.path.join(noises_dir, '*.wav'))
    for class_dir in os.listdir(train_dir):
        print(f"Processing '{class_dir}'")
        wavs_paths = glob.glob(os.path.join(train_dir, class_dir, '*.wav'))
        class_dir_ = os.path.join(output_dir, class_dir)
        check_dir(class_dir_)
        apply_augmentations_dir(wavs_paths, noises_paths, class_dir_, **kwargs)
    
    if create_desc_csv: 
        csv_name = f"desc_train_{kwargs['features_type']}_{kwargs['win_len']}_{kwargs['hop_len']}.csv"
        csv_path = os.path.join(Config.DESC_DIR, csv_name)
        create_desc_file_train(output_dir, csv_path)


def preprocess_test(test_dir, output_dir, create_desc_csv=True, **kwargs):
    """ Extracts features for test data dir """
    print("Extracting features for Test part of dataset.")
    check_dir(output_dir)
    wavs_paths = glob.glob(os.path.join(test_dir, '*.wav'))
    for wav_p in tqdm.tqdm(wavs_paths):
        wav, sr = librosa.load(wav_p, sr=None)
        features = extract_features(
            wav, sr, 
            features_type = kwargs['features_type'], 
            n_components  = kwargs['n_components'], 
            win_len       = kwargs['win_len'],
            hop_len       = kwargs['hop_len'],
            normalize     = kwargs['normalize']
        )
        save_path = os.path.join(output_dir, os.path.basename(wav_p).split('.')[0])
        np.save(save_path, features)
    
    if create_desc_csv:
        csv_name = f"desc_test_{kwargs['features_type']}_{kwargs['win_len']}_{kwargs['hop_len']}.csv"
        csv_path = os.path.join(Config.DESC_DIR, csv_name)
        create_desc_file_test(output_dir, csv_path)


def extract_wav_features(part):
    """ Exctracts acoustix features for specific part of dataset (test or train) """
    extraction_params = {
        'num_augmentations': 1,
        'min_noise_ampl': 0.2,
        'max_noise_ampl': 4.5,
        'features_type': Config.FEATURE_TYPE,
        'n_components': Config.N_COMP,
        'win_len': Config.WIN_LEN,
        'hop_len': Config.HOP_LEN,
        'normalize': True,
    }

    if part == 'train':
        train_dir = os.path.join(Config.DATA_DIR, 'speech_commands_train')
        noises_dir = os.path.join(Config.DATA_DIR, 'noises')
        preprocess_train(
            train_dir, 
            noises_dir, 
            "data/features/{}/train_{}_{}".format(
                extraction_params['features_type'],
                extraction_params['win_len'],
                extraction_params['hop_len']
            ), 
            **extraction_params)

    elif part == 'test':
        test_dir = os.path.join(Config.DATA_DIR, 'speech_commands_test')
        preprocess_test(
            test_dir,
            "data/features/{}/test_{}_{}".format(
                extraction_params['features_type'],
                extraction_params['win_len'],
                extraction_params['hop_len']
            ),
            **extraction_params
        )


def extract_vad_reference_markup():
    print("Extracting reference VAD markup.")
    desc_df = pd.read_csv(os.path.join(Config.DESC_DIR, 'desc_train.csv'))
    extract_and_save_reference_seg(
        desc_df, 
        Config.VAD_REF_DIR,
        win_len    = Config.WIN_LEN, 
        hop_len    = Config.HOP_LEN, 
        frame_size = Config.VAD_FRAME_SIZE
    )
    features_desc_df = pd.read_csv(
        os.path.join(Config.DESC_DIR, 'desc_train_{}_{}_{}.csv'.format(
            Config.FEATURE_TYPE,
            Config.WIN_LEN,
            Config.HOP_LEN
    )))
    create_chunks2aug_features_desc_file(
        features_desc_df, 
        'data/vad_ref_seg', 
        os.path.join(Config.DESC_DIR, 'desc_features2chunks.csv')
    )


def apply_vad(vad_model_path, part, device):
    # Load state dict
    state_dict = torch.load(vad_model_path, map_location='cpu')
    threshold = state_dict['thr']

    # Prepare VAD
    model = UNetVad(Config.VAD_ARCH_SHAPE)
    model.load_state_dict(state_dict['model'])

    desc_df_path = os.path.join(Config.DESC_DIR, f'desc_{part}_mfcc_{Config.WIN_LEN}_{Config.HOP_LEN}.csv')
    desc_df = pd.read_csv(desc_df_path)
    paths = desc_df['wav'].values
    vad_paths = []
    for path in tqdm.tqdm(paths):
        features = torch.Tensor(np.load(path))
        markup = predict_full_overlapped(
            model,
            features    = features, 
            device      = device, 
            frame_size  = Config.VAD_FRAME_SIZE,
            frame_step  = Config.VAD_FRAME_SIZE // 2,
            output_type = 'sigmoid',
            threshold   = threshold, # unused if output_type != 'binary' or 'sigmoid'
            temperature = 1.7        # unused if output_type != 'softmax'
        )
        save_name = '.'.join(os.path.basename(path).split('.')[:-1]) + '.npy'
        save_path = os.path.join(Config.VAD_SYS_DIR, f'{part}', save_name)
        vad_paths.append(save_path)
        np.save(save_path, markup)
    
    # Add 'vad' column in desc_df
    desc_df['vad'] = vad_paths
    desc_df.to_csv(desc_df_path, index=False)


if __name__ == '__main__':
    global_step = 1

    if global_step == 0:
        # Basic preparations
        prepare_project_tree()
        #create_desc_file_train(
        #    os.path.join(Config.DATA_DIR, 'speech_commands_train'),
        #    os.path.join(Config.DESC_DIR, 'desc_train.csv')
        #)
        #create_desc_file_test(
        #    os.path.join(Config.DATA_DIR, 'speech_commands_test'),
        #    os.path.join(Config.DESC_DIR, 'desc_test.csv')
        #)
        extract_wav_features(part='train')
        extract_wav_features(part='test')
        extract_vad_reference_markup()
    elif global_step == 1:
        # After VAD trained: extract vad markups
        apply_vad(vad_model_path='models/vad.pth', part='test', device='cuda:0')
        apply_vad(vad_model_path='models/vad.pth', part='train', device='cuda:0')
    elif global_step == 2:
        # After Extractor and/or Classifier trained: extract embeddings OR make predictions
        pass