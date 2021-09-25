import os
import glob
import tqdm

import numpy as np

import librosa
import soundfile

import torch
import torchaudio


def load_audio(path):
    """ Loads and resamples audio data """
    wav, sr = torchaudio.load(path, normalize=True, channels_first=True)
    if wav.shape[0] > 1: wav = wav.mean(axis=0, keepdim=True)
    return wav


def resample_dir(audio_dir, target_sr):
    """ Resamples all .wav-files in a directory to target_sr """
    wav_paths = glob.glob(os.path.join(audio_dir, '*.wav'))
    for wav_p in tqdm.tqdm(wav_paths):
        wav, sr = librosa.load(wav_p, None)
        if sr != target_sr:
            wav = librosa.resample(wav, sr, target_sr)
        soundfile.write(wav_p, wav, target_sr)


def count_parameters(model):
    """ Counts number of learnable parameters of the model """
    return sum(p.numel() for p in model.parameters())


def measure_inference_timings(model, dummy_tensor, reps=300, device='cuda:0'):
    """ Measures mean inference time and FPS """
    model.to(device)
    dummy_tensor = dummy_tensor.to(device)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = np.zeros([reps, 1])

    # Warming up
    for _ in range(10): model(dummy_tensor)

    # Measuring
    with torch.no_grad():
        for rep in range(reps):
            starter.record()
            model(dummy_tensor)
            ender.record()

            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
        
    mean_inference_time = timings.mean()
    fps = 1000 / mean_inference_time
    return mean_inference_time, fps


def traverse_dir_list(dir_list):
    res = []
    for d in dir_list: res += glob.glob(os.path.join(d, '*.wav'))
    return res


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
