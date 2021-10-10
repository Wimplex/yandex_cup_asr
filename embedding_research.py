import pickle

import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils.inference_utils import extract_embeddings
from utils.data_utils import AudioDataset
from modules.networks import ResNet18_Extractor
from config import Config


def save_embeddings(part, save_path):
    if part == 'train': desc_df = pd.read_csv('desc/desc_train_mfcc_400_160.csv')
    elif part == 'test': desc_df = pd.read_csv('desc/desc_test_mfcc_400_160.csv')
    dataset = AudioDataset(desc_df, classes_list=Config.CLASSES_MAP, frame_size=Config.FRAME_SIZE)
    data_loader = DataLoader(dataset, batch_size=32)

    net_kwargs = {
        'input_shape': Config.NUM_CHANNELS,
        'num_classes': Config.NUM_CLASSES,
        'embed_size':  Config.EMB_SIZE,
    }
    print("Loading model")
    extractor = ResNet18_Extractor(**net_kwargs)
    extractor.load_state_dict(torch.load('models/extractor_loss_2.48.pth'))
    extract_embeddings(extractor, data_loader, device=Config.DEVICE, save_path=save_path)


if __name__ == '__main__':
    extract_embs = False
    part = 'test'
    
    if extract_embs:
        save_embeddings(part, save_path='data/embeddings/%s.pkl' % part)
    else:
        embedding_pairs = pickle.load(open('data/embeddings/%s.pkl' % part, 'rb'))
        embeddings = np.array([p[1] for p in embedding_pairs])
        labels = [p[0] for p in embedding_pairs]
        if part == 'train':
            _, embeddings, _, labels = train_test_split(embeddings, labels, test_size=0.3, stratify=labels, shuffle=True)
        print('Emb array shape:', embeddings.shape)
        umap = UMAP(metric='cosine', low_memory=True, n_neighbors=20)
        emb_reduced = umap.fit_transform(embeddings)

        plt.figure(figsize=(10, 10))
        plt.title("UMAP-reduced %s audio embeddings" % part)
        plt.scatter(emb_reduced[:,0], emb_reduced[:,1], s=1.0, c=labels)
        plt.xlabel("#1 compoenent")
        plt.ylabel("#2 compoenent")
        plt.savefig('img/umap_%s_embeddings.png' % part, dpi=150)
