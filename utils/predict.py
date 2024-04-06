import torch
import pandas as pd
from utils.DataUtils import get_dataloader
from utils.DataUtils import E4Data_freq

def predict(model: object, data: list, y: torch.tensor, device: torch.device, activity_index = None, save_pred: str = '', threshold: float=0.5):
    '''
    Helper function to produce classification prediction data.

    Args:
        model (object): The torch model.
        data (list): List of frequency data.
        y (torch.tensor): Y data.
        device (torch.device): The device the model and data is located on. "CPU", "MPS" (MacBooks with GPU), or "CUDA"
        activity_index (array): An array with indexes for physical activity. Default: None.
        save_pred (str): Name and path for where to save y, y predictions, and logits. Default: '' (not saved).
        threshold (float): The threshold for stress = True. Default: 0.5.

    Returns:
        Y and Y predictions.
    '''
    batch_size = len(y)
    dataloader = get_dataloader(batch_size, shuffle = False, data_class = E4Data_freq(data, device, y))
    for batch_in,labels in dataloader:
        true_prob = torch.sigmoid(model(batch_in)).detach().cpu().numpy()
        #true_prob = model(batch_in)[:,1].detach().cpu().numpy()
        seq_pred = (true_prob > threshold).astype(int)
        
    
    labels = labels.detach().cpu().numpy()

    if save_pred and activity_index is not None:
        pd.DataFrame({'true': labels, 'pred': seq_pred, 'true_prob': true_prob, 'activity': activity_index}).to_pickle(f'{save_pred}.pkl')
    elif save_pred and activity_index is None:
        pd.DataFrame({'true': labels, 'pred': seq_pred, 'true_prob': true_prob}).to_pickle(f'{save_pred}.pkl')
    return labels, seq_pred