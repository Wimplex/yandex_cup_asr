import os
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from config import Config
from networks import DeiT_Extractor, ResNet18_Extractor
from data import AudioDataset, Preprocessor


# TODO: дописать логику тренировочного процесса


def main(model):
    # Setup randomness
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    torch.cuda.manual_seed(Config.SEED)
    torch.backends.cudnn.deterministic = True

    # Training preparations
    torch.cuda.empty_cache()

    # Instantiate model
    net_kwargs = {'input_shape': Config.INPUT_SHAPE, 'num_classes': Config.NUM_CLASSES, 'embed_size': Config.EMB_SIZE}
    if Config.MODEL == 'deit': net_cls = DeiT_Extractor
    elif Config.MODEL == 'resnet18': net_cls = ResNet18_Extractor
    model = net_cls(**net_kwargs)    

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, betas=(1e-4, 0.999), momentum=0.96)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.75, mode='min')
    
    # Create dataloader
    dataset = AudioDataset(
        data_dir=os.path.join(Config.DATA_DIR, 'data'), # <-- исправить, когда будут известны реальные данные
        desc_file_path=os.path.join(Config.DATA_DIR, 'description.csv'), # <-- это тоже
        target_sr=Config.SAMPLE_RATE
    )
    data_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', )
    main()

# TODO "Что не забыть на соревновании":
#   - Понять, как организованы описательные данные (например, description.csv) в предоставленном датасете
#   - Поиграться с аргументом normalize у torchaudio.load() (лежит в utils.py)
#   - Проверить количество всех лейблов в данных
#   - Применить one-hot-кодирование к лейблам.
#   - Посчитать максимальную длительность речи на незашумленных данных