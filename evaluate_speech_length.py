import os
import glob
import random
from pprint import pprint

import pandas as pd

import torch

# Скрипт оценки максимальой длительности речи, 
# необходимой для выяснения ширины окна классификации для главной модели.
# Используется SileroVAD: https://github.com/snakers4/silero-vad#vad


def main():
    # Setup script
    torch.set_num_threads(1)

    # Download SileroVAD model
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero_vad',
        model='silero_vad',
        force_reload=True
    )
    model = model.cpu()

    # Iterate through the data
    wavs_dir = 'path/to/wavs/dir'
    max_len = 0
    wavs_paths = glob.glob(os.path.join(wavs_dir, '*.wav'))
    for wav_path in wavs_paths:
        # wav = load_audio(wav_path, target_sr=16000)
        wav = None

        # CHECK THAT!!
        markup = get_speech_ts(wav, model, num_steps=4)
        if len(markup) > max_len: max_len = markup


if __name__ == '__main__':
    # main()

    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        # force_reload=True
    )
    (get_speech_ts,
    _, _, read_audio,
    _, _, _) = utils

    model = model.cpu()

    df = pd.read_csv('desc/desc_test.csv')
    idx = random.randint(0, len(df))
    wav_p = df.iloc[idx,:]['wav']
    print(wav_p)
    wav = read_audio(wav_p)
    speech_ts = get_speech_ts(wav, model, num_steps=4)
    pprint(speech_ts)
    