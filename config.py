
import torch

class Config:
    # Very obvious comments below

    # Common
    SEED = 3078
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Dir structure
    DATA_DIR = 'data/asr_data/'

    # Data and features preparation params
    SAMPLE_RATE = 16000
    N_FFT = int(0.01 * SAMPLE_RATE)     # FFT win size
    HOP_LEN = int(0.025 * SAMPLE_RATE)  # FFT win step
    FEATURE_TYPE = 'mels'               # Features type
    N_COMP = 64                         # Number of feature components (n_mels / n_mfccs)
    WIN_SIZE = 10                       # Number of time-domain samples of features per input to model

    # Model params
    # MODEL = 'efficientnet-b2'         # Model arch name [deit / resnet18]
    MODEL = 'resnet18'
    EMB_SIZE = 256                      # Embedding size
    NUM_CHANNELS = 3                    # Count of input channels

    # Training params
    BATCH_SIZE = 128
    NUM_EPOCHS_EXTRACTOR = 13
    NUM_EPOCHS_CLASSIFIER = 8
    LEARNING_RATE = 1e-3

    CLASSES_MAP = ['дальше', 'вперед', 'назад', 'вверх', 'вниз', 
                   'выше', 'ниже', 'домой', 'громче', 'тише', 
                   'лайк', 'дизлайк', 'следующий', 'предыдущий',
                   'сначала', 'перемотай', 'выключи', 'стоп', 'хватит',
                   'замолчи', 'заткнись', 'останови', 'пауза', 'включи',
                   'смотреть', 'продолжи', 'играй', 'запусти', 'ноль',
                   'один', 'два', 'три', 'четыре', 'пять', 'шесть',
                   'семь', 'восемь', 'девять']
    NUM_CLASSES = len(CLASSES_MAP)      # Number of classes (number of available in dataset phrases)
