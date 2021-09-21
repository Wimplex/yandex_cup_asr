import os
import sys
sys.path.append(os.path.abspath('..'))

import os
import glob
import torch
from utils.base_utils import load_audio
from utils.vad_utils import get_speech_ts


# Скрипт оценки максимальой длительности речи, 
# необходимой для выяснения ширины окна классификации для главной модели.
# Используется SileroVAD: https://github.com/snakers4/silero-vad#vad


def main():
    # Setup script
    torch.set_num_threads(1)

    # Download SileroVAD model
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero_vad',
        model='silero_vad',
        force_reload=True
    )
    model = model.cpu()

    # Iterate through the data
    wavs_dir = 'path/to/wavs/dir'
    wavs_dir = 'ir'
    max_len = 0
    wavs_paths = glob.glob(os.path.join(wavs_dir, '*.wav'))
    for wav_path in wavs_paths:
        wav = load_audio(wav_path, target_sr=16000)

        # CHECK THAT!!
        markup = get_speech_ts(wav, model, num_steps=4)
        if len(markup) > max_len: max_len = markup



if __name__ == '__main__':
    main()