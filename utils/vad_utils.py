import os
import tqdm
import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from scipy.special import softmax, expit

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utils.base_utils import load_audio

class VadDataset(Dataset):
    def __init__(self, features2chunks_df, mode, label_thr=0.25):
        """ 
            Dataset for vad training purposes 
            :param str ref_seg_path:    Path to reference segmentation .pkl-file
            :param str mode:            Mode of current VAD training.
                                        'classify' for per frame classification, 'segment' for per frame segmentation.
            :param float label_thr:     Decision boundary for converting from seg markup to classification label (with mode == 'classify' only).
                                        If label_thr=0.25 then markup should be filled with or more than 25% of ones to become positive label.
        """
        super(VadDataset, self).__init__()
        self.df = features2chunks_df
        self.mode = mode
        self.label_thr = label_thr

    def split(self, train_size=0.97):
        train_df, test_df = train_test_split(self.df, train_size=train_size, shuffle=True)
        return VadDataset(train_df, self.mode, self.label_thr),\
               VadDataset(test_df, self.mode, self.label_thr)

    def __getitem__(self, idx):
        curr_pair = self.df.iloc[idx,:]
        features = np.load(curr_pair['wav'])
        markup = np.load(curr_pair['markup_path'])
        start, end = os.path.basename(curr_pair['markup_path']).\
                     split('.')[0].split('__')[-1].split('-')
        start, end = int(start), int(end)

        # Get features chunk
        if features.shape[1] < end:
            residual_chunk_size = features.shape[1] % (end - start)
            dummy = np.zeros([features.shape[0], end - start, features.shape[2]])
            dummy[:,:residual_chunk_size,:] = features[:,start:,:]
            features = dummy
        else:
            features = features[:,start:end,:]

        # Converting markup
        if self.mode == 'classify':
            label = np.sum(markup) / len(markup)
            label = int(label > self.label_thr)
        elif self.mode == 'segment':
            label = markup
        return torch.Tensor(features), label

    def __len__(self):
        return len(self.df)


class SileroVadWrap(nn.Module):
    """ 
        Class-wrapper for SileroVAD: https://github.com/snakers4/silero-vad
        :param bool convert_to_frames:  If True, converts outputs from silero-vad to frame-wise markup (level of features frames (mfcc, mels))
        :param int win_len:             Win size of features window
        :param int hop_len:             Step size of features window
        :param float speech_count_thr:  Decision boundary for frames of raw VAD output markup to be speech or non-speech.
                                        Same meaning with VadDataset's label_thr.
    """
    def __init__(self, convert_to_frames=False, win_len=None, hop_len=None, speech_count_thr=0.25):
        super(SileroVadWrap, self).__init__()
        self.convert_to_frames = convert_to_frames
        self.win_len = win_len
        self.hop_len = hop_len
        self.speech_count_thr = speech_count_thr
        self.model, utils_ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            # force_reload=True
        )
        self.model.cpu()
        self.get_speech_ts, _, _, _, _, _, _ = utils_

    @staticmethod
    def timestamps2markup(time_stamps, wav_len):
        """ Converts time-stamps array into torch binary tensor """
        res_tensor = torch.zeros(wav_len)
        for ts in time_stamps: res_tensor[ts['start']:ts['end']] = 1
        return res_tensor

    @staticmethod
    def markup2frames(markup, win_len, hop_len, threshold):
        """ Converts binary VAD-markup into feature-space-like frames """
        if len(markup.shape) == 1: markup = markup.unsqueeze(0).unsqueeze(0)
        return nn.AvgPool1d(kernel_size=win_len, stride=hop_len)(markup) >= threshold
        
    def forward(self, x):
        ts = self.get_speech_ts(x.squeeze(), self.model, num_steps=4)
        markup = self.timestamps2markup(ts, x.shape[1])
        if self.convert_to_frames:
            markup = self.markup2frames(markup, self.win_len, self.hop_len, self.speech_count_thr)
        return markup


def extract_and_save_reference_seg(desc_df, save_dir, win_len, hop_len, frame_size):
    """ 
        Extracts reference VAD markup from wavs, splits into frames and saves it 
        :param str desc_df:     csv-dataframe descrition of wavs dataset
        :param str save_dir:    path to result directory with resulting segmentation
        :param int win_len:     win size of features window
        :param int hop_len:     step size of features window
        :param int frame_size:  count of acoustic features per chunk (per VAD input)
    """
    torch.set_num_threads(4)
    vad = SileroVadWrap(True, win_len, hop_len)
    for wav_path in tqdm.tqdm(desc_df['wav']):
        wav = load_audio(wav_path)
        markup = vad(wav).squeeze()
        frames_count = int(np.ceil(markup.shape[0] / frame_size))
        dummy = np.zeros(shape=[frames_count * frame_size])
        dummy[:markup.shape[0]] = markup
        for i in range(frames_count):
            start, end = i * frame_size, (i + 1) * frame_size
            save_name = '{}__{}-{}.npy'.format(
                os.path.basename(wav_path).split('.')[0], start, end)
            np.save(os.path.join(save_dir, save_name), dummy[start:end])
    torch.set_num_threads(1)


def create_chunks2aug_features_desc_file(features_desc_df, chunks_dir, save_path):
    """ 
        Creates description dataframe with MxN connections 
        between augmented wavs and vad markup chunks.
    """
    chunk_markups = glob.glob(os.path.join(chunks_dir, '*.npy'))
    chunk_markups_list = []
    for m_path in chunk_markups:
        file_name = os.path.basename(m_path).split('.')[0]
        wav_name, edges = file_name.split('__')
        chunk_markups_list.append({'file_name': wav_name, 'markup_path': m_path})
    chunks_df = pd.DataFrame(chunk_markups_list)
    features_desc_df['file_name'] = features_desc_df['wav'].apply(lambda x: os.path.basename(x).split('_')[0])
    result_df = pd.merge(features_desc_df, chunks_df, how='left', on='file_name', validate='many_to_many')
    result_df = result_df.loc[:, ['wav', 'markup_path']]
    result_df.to_csv(save_path, index=False)


def get_longest_speech_segment(vad_markup):
    """ 
        UNUSED.
        Finds and returns start and end indices of first longest speech segment in a VAD markup.
    """
    diff = np.diff(np.r_[0, vad_markup, 0])
    start_ids = np.argwhere(diff == 1).squeeze()
    lengths = np.argwhere(diff == -1).squeeze() - start_ids
    if type(lengths) != np.ndarray:
        start, end = start_ids, start_ids + lengths
    else:
        max_id = np.argmax(lengths)
        start, end = start_ids[max_id], start_ids[max_id] + lengths[max_id]
    return start, end


def centered_sigmoid(x, center):
    return expit(x - center)


def predict_full_overlapped(model, features, device, frame_size, frame_step, output_type, threshold=None, temperature=1.0):
    """ 
        Extracts VAD markup for full feature vector using overlapping frames.
        Use of 'frame_size == frame_step' causes to simple sequence splitting.
        :param model:               Trained PyTorch VAD-model
        :param float threshold:     Best VAD eer_trheshold got on test set
        :param np.ndarray features: N-dim vector of some acoustic features type (mfcc, mels, ...)
        :param str device:          ...
        :param int frame_size:      Input VAD frame size (window size features split to)
        :param int frame_step:      Frame stride/step/hop
        :param str output_type:     Type of output markup vector. One of ['logits', 'sigmoid', 'softmax', 'binary']
        :param float threshold:     Decisions boundary for output_type == 'logits'
        :param float temperature:   Temperature for output_type == 'softmax'
    """
    assert output_type != 'binary' or threshold is not None, "output_type should be 'binary' if threshold is not None"

    model.to(device)
    model.eval()
    features.to(device)
    num_channels = features.shape[0]
    num_components = features.shape[2]
    num_frames = int(np.ceil((features.shape[1] - frame_size) / frame_step))

    # Unfolding func
    def split_tensor_overlapped(tensor, dim):
        unfolded = tensor.unfold(dim, frame_size, frame_step)
        unfolded = unfolded.moveaxis(0, 1).moveaxis(2, 3)
        return unfolded

    # Pad feature matrix with zeros to right and unfold it into strided framess
    input_tensor = torch.zeros([num_channels, num_frames * frame_step + frame_size, num_components]).to(device)
    input_tensor[:, :features.shape[1], :] = features
    frames = split_tensor_overlapped(input_tensor, 1)

    # Get prediction for frames
    vad_frames = torch.empty([frames.shape[0], input_tensor.shape[1]]).fill_(np.nan)
    with torch.no_grad():
        for i, frame in enumerate(frames):
            start = i * frame_step
            end = start + frame_size
            markup = model(frame.unsqueeze(0))
            vad_frames[i, start:end] = markup

    # Compute mean logits per frame and convert it to another representation if needed
    res = np.nanmean(vad_frames.cpu().numpy(), axis=0)
    if output_type == 'binary': res = res > threshold
    elif output_type == 'sigmoid': res = centered_sigmoid(res, threshold)
    elif output_type == 'softmax': res = softmax(res / temperature)
    return res


def post_process_vad_markup(vad_markup_logits, features_hop_size_ms, max_speech_duration_ms=640):
    """ 
        UNUSED.
        Outputs true speech segments according to some settings
        :param np.ndarray vad_markup_logits:    VAD output logits
        :param int feature_hop_size_ms:         Shallow feature extractor hop_size (mfcc, mels, ...) in milliseconds
        :param str process_as:                  Process vad_markup as logits or as probs
        :param int max_speech_duration_ms:      Maximum speech lengths extracted by postprocessor in milliseconds
    """
    speech_indices = vad_markup_logits.argsort()[-int(max_speech_duration_ms // features_hop_size_ms):]
    vad_seg = np.zeros(vad_markup_logits.shape[0])
    vad_seg[speech_indices] = 1
    diff = np.diff(np.r_[0, vad_seg, 0])
    start_ids = np.argwhere(diff == 1).squeeze().astype(np.uint64)
    lengths = np.argwhere(diff == -1).squeeze() - start_ids
    end_ids = (start_ids + lengths).astype(np.uint64)

    if start_ids.ndim == 0:
        start_ids = [start_ids.tolist()]
        end_ids = [end_ids]
    return list(zip(start_ids, end_ids))


#def post_process_vad_markup(vad_markup_logits, features_hop_size_ms, max_speech_duration_ms=640, weight_function='softmax', softmax_temperature=1.0):
#    """
#        Postprocess VAD-markup to improve predictions.
#        Returns start and end points of speech segment.
#        :param np.ndarray vad_markup_logits:    VAD output logits
#        ::
#    """
#
#    # Get most reliable markup elements
#    speech_indices = vad_markup_logits.argsort()[-int(max_speech_duration_ms // features_hop_size_ms):]
#    vad_seg = np.zeros(vad_markup_logits.shape[0])
#    vad_seg[speech_indices] = 1
#
#    # Compute speech segments
#    diff = np.diff(np.r_[0, vad_seg, 0])
#    print(diff)
#    start_ids = np.argwhere(diff == 1).squeeze()
#    lengths = np.argwhere(diff == -1).squeeze() - start_ids
#    end_ids = start_ids + lengths
#
#    if type(end_ids) == np.ndarray:
#        # Compute segments and length weights
#        markup_logits = []
#        for start, end in zip(start_ids, end_ids):
#            # print('\t', vad_markup_logits[start:end].mean())
#            markup_logits.append(vad_markup_logits[start:end].mean())
#
#        markup_weights = softmax(markup_logits)
#        if weight_function == 'softmax': length_weights = softmax(lengths / softmax_temperature)
#        elif weight_function == 'scale': length_weights = lengths / np.max(lengths)
#
#        common_weights = markup_weights * length_weights
#        result_idx = np.argmax(common_weights)
#        print('Markup weights:', markup_weights)
#        print('Length weights:', length_weights)
#        print('Common weights:', common_weights)
#        print("Result idx:", result_idx)
#        start, end = start_ids[result_idx], end_ids[result_idx]
#    else:
#        start, end = start_ids, end_ids
#
#    return start, end