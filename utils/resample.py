import numpy as np
import numpy as np
import pandas as pd 
import scipy 

def ResampleE4Data(BVP, EDA, HR, TEMP, duration: int = 6*60, sampleRate: int = 4):
    # Duration is in seconds
    # Assumption of input: 
        # BVP is 64Hz
        # EDA is 4Hz
        # HR is 1Hz
        # TEMP is 4Hz 

    if sampleRate != 64:
        # Down-sample BVP 
        BVP = scipy.signal.resample(BVP, sampleRate*duration)
    if sampleRate != 4:
        EDA = scipy.signal.resample(EDA, sampleRate*duration)
        x_temp = np.linspace(0, 10, len(TEMP))
        x_temp_re = np.linspace(0, 10, int(len(TEMP)/4*sampleRate))
        # Interpolate upsampled array
        TEMP = np.interp(x_temp_re, x_temp, TEMP)
    if sampleRate != 1:
        x_hr = np.linspace(0, 10, len(HR))
        x_hr_re = np.linspace(0, 10, int(len(HR)*sampleRate))
        # Interpolate upsampled array
        HR = np.interp(x_hr_re, x_hr, HR)
    return BVP, EDA, HR, TEMP

def resample_trainValTest(X_train: list, X_val: list, X_test: list, duration: int = 5*60, sampleRate: int = 4) -> tuple[list, list, list]:
    '''
    Resample training, validation, and test data to a specified sampling rate.

    Args:
        X_train (list): List of training data for the different biosignals.
        X_val (list): List of validation data for the different biosignals.
        X_test (list): List of test data for the different biosignals.
        duration (int): Duration of signal in seconds. Default: 300sec.
        sampleRate (int): Sampling rate of signal in Hz. Default: 4.

    Returns:
        tuple (list, list, list): A tuple with the resampled train, val, and test data.
    '''

    ## Training data
    if X_train:
        B_train_list = []
        E_train_list = []
        H_train_list = []
        T_train_list = []
        for i in range(X_train[1].shape[0]):
            B_train, E_train, H_train, T_train = ResampleE4Data(BVP=X_train[0][i], EDA=X_train[1][i], HR=X_train[2][i], TEMP=X_train[3][i], duration = duration, sampleRate = sampleRate)
            B_train_list.append(B_train)
            E_train_list.append(E_train)
            H_train_list.append(H_train)
            T_train_list.append(T_train)
        ## Convert to np.array
        B_train_array = np.array(B_train_list)
        E_train_array = np.array(E_train_list)
        H_train_array = np.array(H_train_list)
        T_train_array = np.array(T_train_list)
        ## Combine to one array
        X_train_resampled = np.stack([B_train_array, E_train_array, H_train_array, T_train_array], axis=2)
    else:
      X_train_resampled = None

    if X_val:
        ## Validation data
        B_val_list = []
        E_val_list = []
        H_val_list = []
        T_val_list = []
        for i in range(X_val[1].shape[0]):
            B_val, E_val, H_val, T_val = ResampleE4Data(BVP=X_val[0][i], EDA=X_val[1][i], HR=X_val[2][i], TEMP=X_val[3][i], duration = duration, sampleRate = sampleRate)
            B_val_list.append(B_val)
            E_val_list.append(E_val)
            H_val_list.append(H_val)
            T_val_list.append(T_val)
        ## Combine to one array
        B_val_array = np.array(B_val_list)
        E_val_array = np.array(E_val_list)
        H_val_array = np.array(H_val_list)
        T_val_array = np.array(T_val_list)
        ## Combine to one array
        X_val_resampled = np.stack([B_val_array, E_val_array, H_val_array, T_val_array], axis=2)
    else:
      X_val_resampled = None  

    if X_test:
        ## Test data
        B_test_list = []
        E_test_list = []
        H_test_list = []
        T_test_list = []
        for i in range(X_test[1].shape[0]):
            B_test, E_test, H_test, T_test = ResampleE4Data(BVP=X_test[0][i], EDA=X_test[1][i], HR=X_test[2][i], TEMP=X_test[3][i], duration = duration, sampleRate = sampleRate)
            B_test_list.append(B_test)
            E_test_list.append(E_test)
            H_test_list.append(H_test)
            T_test_list.append(T_test)
        ## Convert to np.array
        B_test_array = np.array(B_test_list)
        E_test_array = np.array(E_test_list)
        H_test_array = np.array(H_test_list)
        T_test_array = np.array(T_test_list)
        ## Combine to one array
        X_test_resampled = np.stack([B_test_array, E_test_array, H_test_array, T_test_array], axis=2)
    else:
        X_test_resampled = None

    return X_train_resampled, X_val_resampled, X_test_resampled

def resample_trainValTest_single(X_data: list, duration: int = 5*60, sampleRate: int = 4) -> list:
    '''
    Resample training, validation, and test data to a specified sampling rate.

    Args:
        X_data (list): List of data for the different biosignals.
        duration (int): Duration of signal in seconds. Default: 300sec.
        sampleRate (int): Sampling rate of signal in Hz. Default: 4.

    Returns:
        list: A list with the resampled data.
    '''

    B_list = []
    E_list = []
    H_list = []
    T_list = []
    for i in range(X_data[1].shape[0]):
        B_train, E_train, H_train, T_train = ResampleE4Data(BVP=X_data[0][i], EDA=X_data[1][i], HR=X_data[2][i], TEMP=X_data[3][i], duration = duration, sampleRate = sampleRate)
        B_list.append(B_train)
        E_list.append(E_train)
        H_list.append(H_train)
        T_list.append(T_train)
    ## Convert to np.array
    B_array = np.array(B_list)
    E_array = np.array(E_list)
    H_array = np.array(H_list)
    T_array = np.array(T_list)
    ## Combine to one array
    X_resampled = np.stack([B_array, E_array, H_array, T_array], axis=2)

    return X_resampled

