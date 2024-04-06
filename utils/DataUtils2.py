from typing import List
from torch.utils.data import DataLoader
import torch
from utils.resample import resample_trainValTest, resample_trainValTest_single


class E4Data_freq(torch.utils.data.Dataset):

    def __init__(self, data: List, device: torch.device, data_op=None):
        '''
        Initialize the E4Data_freq class.

        Args:
            data (List): The X data to be loaded.
            device (torch.device): The device to be used.
            data_op (torch.tensor, List): The target data to be loaded.
        '''
        self.data = data
        self.labels = data_op
        self.device = torch.device(device)

    def __len__(self) -> int:
        '''
        Get the length of the data.

        Returns:
            int: The length of the dataset.
        '''
        return self.data[0].shape[0]
    
    def __str__(self) -> str:
        '''
        Get a string representation of the E4Data object.

        Returns:
            str: A string representation of the E4Data object.
        '''
        return f"E4Data_freq object with {len(self.data)} items"

    def __getitem__(self, index):
        '''
        Get an item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            list: The item at the specified index.
        '''
        try:
            out = [d[index, :, :].to(self.device) for d in self.data]
            if self.labels is None:
                return out
            else:
                return out, self.labels[index].to(self.device)
        except IndexError:
            raise IndexError("Invalid index")
    
class E4Data(torch.utils.data.Dataset):
    def __init__(self, data: List, device: torch.device):
        '''
        Initialize the E4Data class.

        Args:
            data (List): The X data to be loaded.
            device (torch.device): The device to be used.
        '''
        self.data = data
        self.device = torch.device(device)

    def __len__(self) -> int:
        '''
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        '''
        return self.data[0].shape[0]

    def __str__(self) -> str:
        '''
        Get a string representation of the E4Data object.

        Returns:
            str: A string representation of the E4Data object.
        '''
        return f"E4Data object with {len(self)} items"
    
    def __getitem__(self, index: int):
        '''
        Get an item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            list: The item at the specified index.
        '''
        try:
            out = [d[index, :].to(self.device) for d in self.data]
            return out
        except IndexError:
            raise IndexError("Invalid index")

def get_dataloader(batch_size: int, shuffle: bool = False, drop_last: bool = False, data_class = E4Data_freq) -> DataLoader:
    """
    Creates dataloader for X frequency data. This is specifically the resampled data.

    Args:
        data (list): List of data for the different biosignals for the different sampling rates. Dim: [number of subject, duration*sampling rate, number of biosignals].
        data_out (str): The output data.
        batch_size (int): Batch size for each epoch.
        shuffle (boolean): If the data should be randomly shuffled. Default: ``False``.
        drop_last (boolean): If the last rows/data should be dropped if length is unequal to batch size. Default: ``False``.
        data_class (class): The class for the data. Default: ``E4Data_freq``.

    Returns:
        torch.utils.data.DataLoader: The dataloader object.

    """

    #e4 = E4Data_freq(data, device, data_out)
    data_loader = DataLoader(
        data_class,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return data_loader

def get_freq(X_train: list, X_val: list, X_test: list, duration: int) -> tuple[list, list, list]:
    """ 
    Returns data resampled to 4Hz and 64Hz (data with these sampling rates are not resampled). 
    The dimensions of the return data is a list with two arrays. Where each array has dim: [number of subjects, duration*sampling rate, number of biosignals].

    Args:
        X_train (list): List of training data for the different biosignals.
        X_val (list): List of validation data for the different biosignals.
        X_test (list): List of test data for the different biosignals.
        duration (int): Duration of signal in seconds. Default: 300.
    
    Returns:
        tuple (list, list, list): A tuple with the data resampled to 4Hz and 64Hz.
    """
    
    X_train_4, X_val_4, X_test_4 = resample_trainValTest(X_train, X_val, X_test, duration, sampleRate=4)
    X_train_64, X_val_64, X_test_64 = resample_trainValTest(X_train, X_val, X_test, duration, sampleRate=64)
    # if torch.backends.mps.is_available():
    X_train_freq = [torch.tensor(X_train_4).float(), torch.tensor(X_train_64).float()]
    X_val_freq = [torch.tensor(X_val_4).float(), torch.tensor(X_val_64).float()]
    X_test_freq = [torch.tensor(X_test_4).float(), torch.tensor(X_test_64).float()]
    # else:
        # X_train_freq = [torch.tensor(X_train_4), torch.tensor(X_train_64)]
        # X_val_freq = [torch.tensor(X_val_4), torch.tensor(X_val_64)]
        # X_test_freq = [torch.tensor(X_test_4), torch.tensor(X_test_64)]
    
    return X_train_freq, X_val_freq, X_test_freq

def convert_X_tensor(X_train, X_val, X_test) -> tuple[list, list, list]:
    '''
    Converts the X data to tensor.

    Args:
        X_train (list): List of training data for the different biosignals.
        X_val (list): List of validation data for the different biosignals.
        X_test (list): List of test data for the different biosignals.
    
    Returns:
        tuple (list, list, list): A tuple with lists of data converted to tensor.
    '''
    # if torch.backends.mps.is_available():
    X_train = [torch.tensor(i).float() for i in X_train]
    X_val = [torch.tensor(i).float() for i in X_val]
    X_test = [torch.tensor(i).float() for i in X_test]
    # else:
        # X_train = [torch.tensor(i) for i in X_train]
        # X_val = [torch.tensor(i) for i in X_val]
        # X_test = [torch.tensor(i) for i in X_test]
    return X_train, X_val, X_test

def get_freq_single(X_data: list, duration: int) -> list:
    """ 
    Returns data resampled to 4Hz and 64Hz (data with these sampling rate are not resampled). 
    The dimensions of the return data is a list with two arrays. Where each array has dim: [number of subject, duration*sampling rate, number of biosignals].

    Args:
        X_data (list): List of data for the different biosignals.
        duration (int): Duration of signal in seconds. Default: 300.
    
    Returns:
        list: the data resampled for one data stream (training, val, or test).
    """
    
    X_data_4 = resample_trainValTest_single(X_data, duration, sampleRate=4)
    X_data_64 = resample_trainValTest_single(X_data, duration, sampleRate=64)
    # if torch.backends.mps.is_available():
    X_data_freq = [torch.tensor(X_data_4), torch.tensor(X_data_64).float()]
    # else:
        # X_data_freq = [torch.tensor(X_data_4), torch.tensor(X_data_64)]
    
    return X_data_freq

def convert_X_tensor_single(X_data: list) -> list:
    '''
    Converts the X data to tensor.

    Args:
        X_data (list): List of data for the different biosignals.

    Returns:
        list: The data converted to tensor for one data stream (training, val, or test).
    '''
    # if torch.backends.mps.is_available():
    X_data = [torch.tensor(i).float() for i in X_data]
    # else:
        # X_data = [torch.tensor(i) for i in X_data]
    
    return X_data




#resample_trainValTest_single