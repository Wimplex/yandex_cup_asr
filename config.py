
import torch

class Config:
    # Quite obvious comments below

    # Common
    SEED = 451
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Data and features preparation params
    SAMPLE_RATE = 16000
    WIN_LEN = int(0.025 * SAMPLE_RATE)  # FFT win size
    HOP_LEN = int(0.01 * SAMPLE_RATE)   # FFT win step
    FEATURE_TYPE = 'mfcc'               # Features type
    N_COMP = 23                         # Number of feature components (n_mels / n_mfccs)
    FRAME_SIZE = 224                    # Number of time-domain samples of features per input to Extractor model
    VAD_FRAME_SIZE = 64                 # Number of time-domain samples of features per input to VAD model

    # Data dirs structure
      # Already have
    DATA_DIR = 'data/asr_data/'
    NOISES_DIR = 'data/asr_data/noises'
      # To create
    FEATURES_DIR = f'data/features/{FEATURE_TYPE}'
    VAD_REF_DIR = 'data/vad_ref_seg'
    VAD_SYS_DIR = 'data/features/vad'
    EMBEDDINGS_DIR = 'data/embeddings'
    DESC_DIR = 'desc'

    # Extractor model params
    MODEL = 'resnet18'                  # Model arch name [deit / resnet18 / efficientnet-bX]
    EMB_SIZE = 512                      # Embedding size
    NUM_CHANNELS = 3                    # Count of input channels

    # VAD model params
    VAD_ARCH_SHAPE = [16, 32, 64, 128,  # Number of channels in UNet blocks
                      256,
                      128, 64, 32, 16]
    VAD_LOSS_TYPE = 'dice'

    # Training params
    BATCH_SIZE = 128
    NUM_EPOCHS_EXTRACTOR = 10
    NUM_EPOCHS_CLASSIFIER = 4
    NUM_EPOCHS_VAD = 1
    LEARNING_RATE = 3e-3
    EVAL_EVERY = 500

    # Classes
    CLASSES_MAP = ['дальше', 'вперед', 'назад', 'вверх', 'вниз', 
                   'выше', 'ниже', 'домой', 'громче', 'тише', 
                   'лайк', 'дизлайк', 'следующий', 'предыдущий',
                   'сначала', 'перемотай', 'выключи', 'стоп', 'хватит',
                   'замолчи', 'заткнись', 'останови', 'пауза', 'включи',
                   'смотреть', 'продолжи', 'играй', 'запусти', 'ноль',
                   'один', 'два', 'три', 'четыре', 'пять', 'шесть',
                   'семь', 'восемь', 'девять']
    NUM_CLASSES = len(CLASSES_MAP)
