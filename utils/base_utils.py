import os
import glob
import numpy as np
import torch
import torchaudio


def load_audio(path, target_sr):
    """ Loads and resamples audio data """
    wav, sr = torchaudio.load(path, normalize=True, channels_first=True)
    if wav.shape[0] > 1: wav = wav.mean(axis=0, keepdim=True)
    if sr != target_sr: torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


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
