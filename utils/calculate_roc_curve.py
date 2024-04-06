import torch
import numpy as np
from sklearn.metrics import roc_curve, average_precision_score, precision_recall_curve
from utils.DataUtils import get_dataloader

def roc(model, data: list, y: torch.tensor, device: torch.device, shuffle = False, save_pred = ''):
    '''
    Helper function to produce classification prediction data.

    Args:
        model: The torch model.
        data (list): List of frequency data.
        y (torch.tensor): Y data.
        device (torch.device): The device the model and data is located on. "CPU", "MPS" (MacBooks with GPU), or "CUDA"
        shuffle (bool): If data should be shuffled. Default ``False``.
    '''
    batch_size = len(y)
    dataloader = get_dataloader(data, y, batch_size, device, shuffle = shuffle)
    for batch_in,labels in dataloader:
        true_prob = model(batch_in)[:,1].detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

    fpr, tpr, thresholds = roc_curve(labels,true_prob)
    return fpr, tpr, thresholds 

def aps(model, data: list, y: torch.tensor, device: torch.device, shuffle = False, save_pred = ''):
    '''
    Helper function to produce classification prediction data.

    Args:
        model: The torch model.
        data (list): List of frequency data.
        y (torch.tensor): Y data.
        device (torch.device): The device the model and data is located on. "CPU", "MPS" (MacBooks with GPU), or "CUDA"
        shuffle (bool): If data should be shuffled. Default ``False``.
    '''
    batch_size = len(y)
    dataloader = get_dataloader(data, y, batch_size, device, shuffle = shuffle)
    for batch_in,labels in dataloader:
        true_prob = model(batch_in)[:,1].detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

    return average_precision_score(labels,true_prob)

def pr(model, data: list, y: torch.tensor, device: torch.device, shuffle: bool = False, save_pred = ''):
    '''
    Helper function to produce classification prediction data.

    Args:
        model: The torch model.
        data (list): List of frequency data.
        y (torch.tensor): Y data.
        device (torch.device): The device the model and data is located on. "CPU", "MPS" (MacBooks with GPU), or "CUDA"
        shuffle (bool): If data should be shuffled. Default ``False``.
    '''
    batch_size = len(y)
    dataloader = get_dataloader(data, y, batch_size, device, shuffle = shuffle)
    for batch_in,labels in dataloader:
        true_prob = model(batch_in)[:,1].detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

    
    return precision_recall_curve(labels, true_prob)