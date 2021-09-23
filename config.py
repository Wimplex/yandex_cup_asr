
class Config:
    # Very obvious comments below

    # Common
    SEED = 42

    # Dir structure
    DATA_DIR = 'data/asr_data/'
    NOISES_DIR = ['data/asr_data/noises', 'data/noises']
    IR_DIR = ['data/ir/ir_set1', 'data/ir/ir_set2']

    # Data and features preparation params
    SAMPLE_RATE = 16000
    N_FFT = int(0.01 * SAMPLE_RATE)     # FFT win size
    HOP_LEN = int(0.025 * SAMPLE_RATE)  # FFT win step
    FEATURE_TYPE = 'mels'               # Features type
    N_COMP = 64                         # Number of feature components (n_mels / n_mfccs)
    WIN_SIZE = 10                       # Number of time-domain samples of features per input to model
    NUM_CLASSES = 38                    # Number of classes (number of available in dataset phrases)

    # Model params
    MODEL = 'resnet18'                  # Model arch name [deit / resnet18]
    EMB_SIZE = 256                      # Embedding size
    INPUT_SHAPE = [6, WIN_SIZE, N_COMP] # Shape of input tensors

    # Training params
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3
