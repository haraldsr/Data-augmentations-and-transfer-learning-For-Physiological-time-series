from dataclasses import dataclass
from typing import Tuple
import torch
import numpy as np
import torch

@dataclass
class model_config:
    '''
    Class with all configurations for the model.

    Variables:
        Numcops (int): Number of components in the model which is equal to the number of frequencies included.
        Numfeats (list): Number of signals per frequency. Relevant if you include subset of signal for a given frequency.
        nodes_before_last_layer (int): Number of nodes before last layer.
        nodes_per_freq (list): Number of "out" nodes for each frequency in encoder layer.
        dropout (float): The dropout rate for each layer.
        architecture (str): What model architecture to use.
        NumLayers (int): The number of layers in the encoder part.
        CNNKernelSize (list): The kernel sizes for each layer.
        Out_Nodes (int): The number of nodes in the out layers.
        seq_length (int): The length of the sequence for each frequency. 
        R_U (bool): If the CNN layer should be a reverse U-Net architecture.
    '''
    NumComps: int
    NumFeats: list
    nodes_before_last_layer: int
    nodes_per_freq: list
    dropout: float
    architecture: str
    NumLayers: int
    CNNKernelSize: list
    Out_Nodes: int
    seq_length: int
    R_U: bool

@dataclass
class auto_config:
    '''
    Class with all configurations for training autoencoder.

    Variables:
        model (object): The model to be trained.
        model_name (str): The name that the model should be saved as.
        device (torch.device): The device where data, and model is located.
        data (tuple(list,list,list)): The X data.
        original_data (tuple(list,list,list)): The original data to reconstruct.
        n_epochs (int): Number of epochs.
        batch_size (int): The batch size.
        lr (float): Learning rate.
        patience (int): Patience, the number of epochs with non-improved validation loss before early stop.
        delta (float): The relative improvement in validation loss. (Also applies to lr scheduler).
        optimizer_name (str): Name of the optimizer to be used.
        lr_scheduler_name (str): The name of the learning rate scheduler to be used.
        criterion_name (str): The name of the loss function to be used.
        seed (int): The random seed to be used.
        shuffle (bool). If the data in the dataloader should be shuffled.
        path_models (str): Path where the models should be saved.
        folder_runs (str): The folder where plots and run information should be saved.
        txt_output (str): The txt file/str with run information.
        run (int): The run number.
    '''
        
    model: object
    model_name: str
    device: torch.device
    data: Tuple[list, list, list]
    original_data: Tuple[list, list, list]
    n_epochs: int
    batch_size: int
    lr: float
    patience: int
    delta: float
    optimizer_name: str
    lr_scheduler_name: str
    criterion_name: str
    seed: int
    shuffle: bool
    path_models: str
    folder_runs: str
    txt_output: str
    run: int

    def optimizer(self):
        return get_optimizer(self)
    def lr_scheduler(self, optimizer):
        return get_lr_scheduler(self, optimizer)
    def criterion(self, weights=None):
        return get_criterion(self, weights)

@dataclass
class classi_config:
    '''
    Class with all configurations for training autoencoder.

    Variables:
        model (object): The model to be trained.
        model_name (str): The name that the model should be saved as.
        device (torch.device): The device where data, and model is located.
        data (tuple(list,list,list)): The X data.
        y_data (tuple(torch.tensor, torch.tensor,torch.tensor)): The stress labels.
        data_class (object): The class to use for the dataloader
        n_epochs (int): Number of epochs.
        batch_size (int): The batch size.
        lr (float): Learning rate.
        patience (int): Patience, the number of epochs with non-improved validation loss before early stop.
        delta (float): The relative improvement in validation loss. (Also applies to lr scheduler).
        optimizer_name (str): Name of the optimizer to be used.
        lr_scheduler_name (str): The name of the learning rate scheduler to be used.
        criterion_name (str): The name of the loss function to be used.
        seed (int): The random seed to be used.
        shuffle (bool). If the data in the dataloader should be shuffled.
        path_models (str): Path where the models should be saved.
        folder_runs (str): The folder where plots and run information should be saved.
        txt_output (str): The txt file/str with run information.
        run (int): The run number.
    '''

    model: object
    model_name: str
    device: torch.device
    data: Tuple[list, list, list]
    y_data: Tuple[torch.tensor, torch.tensor,torch.tensor]
    data_class: object
    n_epochs: int
    batch_size: int
    lr: int
    patience: int
    delta: float
    optimizer_name: str
    lr_scheduler_name: str
    criterion_name: str
    seed: int
    shuffle: bool
    path_models: str
    folder_runs: str
    txt_output: str
    run: int

    def optimizer(self):
        return get_optimizer(self)
    def lr_scheduler(self, optimizer):
        return get_lr_scheduler(self, optimizer)
    def criterion(self, weights=None):
        return get_criterion(self, weights)

@dataclass
class sub_diff_config:
    '''
    Class with all configurations for training autoencoder.

    Variables:
        model (object): The model to be trained.
        model_name (str): The name that the model should be saved as.
        device (torch.device): The device where data, and model is located.
        data (tuple(list,list,list)): The X data.
        y_data (tuple(torch.tensor, torch.tensor,torch.tensor)): The subject ID labels.
        n_epochs (int): Number of epochs.
        batch_size (int): The batch size.
        lr (float): Learning rate.
        patience (int): Patience, the number of epochs with non-improved validation loss before early stop.
        delta (float): The relative improvement in validation loss. (Also applies to lr scheduler).
        optimizer_name (str): Name of the optimizer to be used.
        lr_scheduler_name (str): The name of the learning rate scheduler to be used.
        criterion_name (str): The name of the loss function to be used.
        seed (int): The random seed to be used.
        shuffle (bool). If the data in the dataloader should be shuffled.
        path_models (str): Path where the models should be saved.
        folder_runs (str): The folder where plots and run information should be saved.
        run (int): The run number.
    '''
    model: torch.nn.Module
    model_name: str
    device: torch.device
    data: Tuple[list, list, list]
    y_data: Tuple[torch.tensor, torch.tensor,torch.tensor]
    n_epochs: int
    batch_size: int
    lr: int
    patience: int
    delta: float
    optimizer_name: str
    lr_scheduler_name: str
    criterion_name: str
    seed: int
    shuffle: bool
    path_models: str
    folder_runs: str
    run: int

    def optimizer(self):
        return get_optimizer(self)
    def lr_scheduler(self, optimizer):
        return get_lr_scheduler(self, optimizer)
    def criterion(self, weights=None):
        return get_criterion(self, weights)
    
def get_optimizer(config: object) -> torch.optim:
    ''''
    Args:
        The config for the instance.

    Returns:
        torch.optim: The optimizer.
    '''
    if config.model is None:
        raise ValueError("Model is not specified.")
    
    if config.optimizer_name == 'SGD':
        return torch.optim.SGD(config.model.parameters(), lr=config.lr,momentum=0.9)
    elif config.optimizer_name == 'Adam':
        return torch.optim.Adam(config.model.parameters(), lr=config.lr, weight_decay=1e-04) 
    elif config.optimizer_name == 'NAdam':
        return torch.optim.NAdam(config.model.parameters(), lr=config.lr)
    elif config.optimizer_name == 'AMSGrad':
        return torch.optim.Adam(config.model.parameters(), lr=config.lr, amsgrad=True)
    else:
        raise ValueError(f'Unsupported optimizer: {config.optimizer}.')
    
def get_lr_scheduler(config: object, optimizer: torch.optim) -> torch.optim.lr_scheduler:
    ''''
    Args:
        The config for the instance.

    Returns:
        torch.lr_scheduler: The learning rate scheduler.
    '''
    if config.optimizer is None:
        raise ValueError("Optimizer is not specified.")
    if config.lr_scheduler_name == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, threshold=config.delta, threshold_mode='rel', patience=5)
    elif config.lr_scheduler_name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9) 
    elif config.lr_scheduler_name == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif config.lr_scheduler_name == 'LambdaLR':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9**epoch)
    elif config.lr_scheduler_name is None:
        return None
    else:
        raise ValueError(f'Unsupported learning scheduler: {config.lr_scheduler}.')
    
def get_criterion(config: object, weights=None) -> torch.nn:
    ''''
    Args:
        The config for the instance.

    Returns:
        torch.nn: The loss function.
    '''
    if config.criterion_name == 'MSELoss':
        return torch.nn.MSELoss()
    elif config.criterion_name == 'L1Loss':
        return torch.nn.L1Loss()
    elif config.criterion_name == 'BCEWithLogitsLoss': #binary classification
        return torch.nn.BCEWithLogitsLoss(pos_weight=weights)
    elif config.criterion_name == 'CrossEntropyLoss': #classification
        return torch.nn.CrossEntropyLoss(weight=weights).to(config.device)
    elif config.criterion_name == 'BYOLoss':
        return BYOLoss()
    elif config.criterion_name == 'AMSGrad':
        return torch.optim.Adam(config.model.parameters(), lr=config.lr, amsgrad=True)
    else:
        raise ValueError(f'Unsupported optimizer: {config.optimizer}.')

class BYOLoss(torch.nn.Module):
    def __init__(self):
        super(BYOLoss, self).__init__()

    def forward(self, z1, z2):
        z1 = torch.nn.functional.normalize(z1, dim=-1, p=2)
        z2 = torch.nn.functional.normalize(z2, dim=-1, p=2)
        loss = 2 - 2 * (z1 * z2).sum(dim=-1)
        return loss.mean()