import tqdm
import pickle
import numpy as np
import torch


def evaluate_loss(model, test_loader, device, criterion='model'):
    """ Returns average loss on test set for input model """
    losses = []
    for batch_X, batch_y, batch_vad, _ in tqdm.tqdm(test_loader):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        batch_vad = batch_vad.to(device)
        if criterion == 'model': loss = model(batch_X, batch_vad, batch_y)
        else:
            output_batch = model(batch_X, batch_vad)
            loss = criterion(output_batch, batch_y)
        losses.append(loss.item())
    return np.mean(losses)
        

def predict_classifier(model, data_loader, device):
    """ Returns predictions from classifier model """
    is_model_in_train_mode = model.training
    model = model.to(device)
    if is_model_in_train_mode: model.eval()
    with torch.no_grad():
        pred = torch.Tensor().to(device)
        true = torch.Tensor().to(device)
        names = []
        for batch_X, batch_y, batch_name in tqdm.tqdm(data_loader, total=len(data_loader)):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_out = model(batch_X)
            pred = torch.cat([pred, batch_out.argmax(axis=1)])
            true = torch.cat([true, batch_y])
            names += batch_name
    if is_model_in_train_mode: model.train()
    return pred.cpu().int(), true.cpu(), names


def extract_embeddings(extractor, data_loader, device, save_path=None):
    """ Extracts embeddings from input data """
    if extractor.device != device: extractor = extractor.to(device)
    is_model_in_train_mode = extractor.training
    if is_model_in_train_mode: extractor.eval()
    with torch.no_grad():
        embeds = torch.Tensor().to(device)
        labels = torch.Tensor().to(device)
        for batch_X, batch_y, _ in tqdm.tqdm(data_loader, total=len(data_loader)):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_out = extractor.forward_emb(batch_X)
            embeds = torch.cat([embeds, batch_out])
            labels = torch.cat([labels, batch_y])
    if is_model_in_train_mode: extractor.train()

    if save_path is not None:
        embeds_ = embeds.detach().cpu().numpy()
        labels_ = labels.detach().cpu().numpy()
        pickle.dump(list(zip(labels_, embeds_)), open(save_path, 'wb'))
    return embeds
