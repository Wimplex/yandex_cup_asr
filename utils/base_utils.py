import os
import datetime

import torch
import torchaudio


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def now():
    return datetime.datetime.now().strftime('%m-%d-%H-%M-%S')


def load_audio(path):
    """ Loads and resamples audio data """
    wav, _ = torchaudio.load(path, normalize=True, channels_first=True)
    if wav.shape[0] > 1: wav = wav.mean(axis=0, keepdim=True)
    return wav


def save_model(model, save_path):
    """ What is the purpose of that smol function?.. """
    if type(model) != dict:
        torch.save(model.state_dict(), save_path)
    else:
        torch.save(model, save_path)


def save_checkpoint(model, optimizer, save_path):
    """ Saves checkpoint for future training restart """
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state_dict, save_path)

