### activity_class.py
import numpy as np
import pickle 
import pandas as pd
from scipy.fft import rfft, rfftfreq

def activity_class_std(index_train: int, index_val: int, index_test: int, data_set: str = 'WA', duration: int = 300, path: str = "../Data/", before: int = 0, threshold: list = [40], seed: int = 123):
    ## Read data
    folder = f"{data_set}_{duration}sec{'_bf'+str(before) if before else ''}/"
    file_str = f"{data_set}_{duration}sec{'_bf'+str(before) if before else ''}"
    with open(f'{path}{folder}{file_str}_ACCx.pkl', 'rb') as f:
        accx = pickle.load(f)

    with open(f'{path}{folder}{file_str}_ACCy.pkl', 'rb') as f:
        accy = pickle.load(f)

    with open(f'{path}{folder}{file_str}_ACCz.pkl', 'rb') as f:
        accz = pickle.load(f)

    if data_set == 'ADARP':
        # Getting stress data                
        amount_stress = len(accx[accx['y_stress']==1])
        stress_idx = accx.index[accx['y_stress']==1]
        nostress_idx = accx[accx['y_stress']==0].sample(n = (amount_stress*2), random_state=seed).index                                                      
        accx_stress = accx.iloc[stress_idx,:]
        accy_stress = accy.iloc[stress_idx,:]
        accz_stress = accz.iloc[stress_idx,:]
        # Getting no stress data
        accx_nostress = accx.iloc[nostress_idx,:]
        accy_nostress = accy.iloc[nostress_idx,:]
        accz_nostress = accz.iloc[nostress_idx,:]
        # Combine
        accx = pd.concat([accx_stress, accx_nostress])
        accy = pd.concat([accy_stress, accy_nostress])
        accz = pd.concat([accz_stress, accz_nostress])
    
    accx.drop(['subject', 'y_stress'], inplace=True, axis=1)
    accy.drop(['subject', 'y_stress'], inplace=True, axis=1)
    accz.drop(['subject', 'y_stress'], inplace=True, axis=1)

    acc = np.stack([accx, accy, accz], axis=2)

    ## Split and reorder as physiological data from input indices 
    acc_train = acc[index_train]
    acc_val = acc[index_val]
    acc_test= acc[index_test]

    sd_threshold = threshold[0]

    ## Predict
    # Train
    y_train = []
    for i in range(acc_train.shape[0]):
        if(np.std(acc_train[i,:,:]) >= sd_threshold):
            y_train.append(1)
        else:
            y_train.append(0)

    # Val
    y_val = []
    for i in range(acc_val.shape[0]):
        if(np.std(acc_val[i,:,:]) >= sd_threshold):
            y_val.append(1)
        else:
            y_val.append(0)

    # Test
    y_test = []
    for i in range(acc_test.shape[0]):
        if(np.std(acc_test[i,:,:]) >= sd_threshold):
            y_test.append(1)
        else:
            y_test.append(0)

    return y_train, y_val, y_test

def activity_class_frequency(index_train: list, index_val: list, index_test: list, data_set: str = 'WA', duration: int = 300, path: str = "../Data/", before: int = 0, threshold: list = [2,3], SAMPLE_RATE: int = 32, seed: int = 123):
    ## Read data
    folder = f"{data_set}_{duration}sec{'_bf'+str(before) if before else ''}/"
    file_str = f"{data_set}_{duration}sec{'_bf'+str(before) if before else ''}"
    with open(f'{path}{folder}{file_str}_ACCx.pkl', 'rb') as f:
        accx = pickle.load(f)
    #accx.drop(['subject', 'y_stress'], inplace=True, axis=1)
    with open(f'{path}{folder}{file_str}_ACCy.pkl', 'rb') as f:
        accy = pickle.load(f)
    accy.drop(['subject', 'y_stress'], inplace=True, axis=1)
    with open(f'{path}{folder}{file_str}_ACCz.pkl', 'rb') as f:
        accz = pickle.load(f)
    accz.drop(['subject', 'y_stress'], inplace=True, axis=1)

    if data_set == 'ADARP':
        # Getting stress data                
        amount_stress = len(accx[accx['y_stress']==1])
        stress_idx = accx.index[accx['y_stress']==1]
        nostress_idx = accx[accx['y_stress']==0].sample(n = (amount_stress*2), random_state=seed).index  
        accx.drop(['subject', 'y_stress'], inplace=True, axis=1)                                                           
        accx_stress = accx.iloc[stress_idx,:]
        accy_stress = accy.iloc[stress_idx,:]
        accz_stress = accz.iloc[stress_idx,:]
        # Getting no stress data
        accx_nostress = accx.iloc[nostress_idx,:]
        accy_nostress = accy.iloc[nostress_idx,:]
        accz_nostress = accz.iloc[nostress_idx,:]
        # Combine
        accx = pd.concat([accx_stress, accx_nostress])
        accy = pd.concat([accy_stress, accy_nostress])
        accz = pd.concat([accz_stress, accz_nostress])
    accx.drop(['subject', 'y_stress'], inplace=True, axis=1)

    acc = np.stack([accx, accy, accz], axis=2)

    ## Split and reorder as physiological data from input indices 
    acc_train = acc[index_train]
    acc_val = acc[index_val]
    acc_test= acc[index_test]

    N = SAMPLE_RATE * duration

    ## Predict
    # Train
    y_train = []
    for i in range(acc_train.shape[0]):
        signal_x = acc_train[i,:,0]
        yf = rfft(signal_x)
        xf = rfftfreq(N, 1 / SAMPLE_RATE)
        notAdded = True
        max_index = -1
        while notAdded:
            if abs(max_index) == len(yf):
                index_used = np.where(yf == np.sort(yf)[-1])[0]
                signal_f = xf[index_used][0]
                notAdded = False
            if xf[yf==np.sort(yf)[max_index]][0] < 0.5:
                max_index -= 1
            else:
                index_used = np.where(yf == np.sort(yf)[max_index])[0]
                signal_f = xf[index_used][0]
                notAdded = False

        if (signal_f >= threshold[0]) and (signal_f <=threshold[1]):
            y_train.append(1)
        else:
            y_train.append(0)  

    # Val
    y_val = []
    for i in range(acc_val.shape[0]):
        signal_x = acc_val[i,:,0]
        yf = rfft(signal_x)
        xf = rfftfreq(N, 1 / SAMPLE_RATE)
        notAdded = True
        max_index = -1
        while notAdded:
            if abs(max_index) == len(yf):
                index_used = np.where(yf == np.sort(yf)[-1])[0]
                signal_f = xf[index_used][0]
                notAdded = False
            if xf[yf==np.sort(yf)[max_index]][0] < 0.5:
                max_index -= 1
            else:
                index_used = np.where(yf == np.sort(yf)[max_index])[0]
                signal_f = xf[index_used][0]
                notAdded = False

        if (signal_f >= threshold[0]) and (signal_f <=threshold[1]):
            y_val.append(1)
        else:
            y_val.append(0)  

    # Test
    y_test = []
    for i in range(acc_test.shape[0]):
        signal_x = acc_test[i,:,0]
        yf = rfft(signal_x)
        xf = rfftfreq(N, 1 / SAMPLE_RATE)
        notAdded = True
        max_index = -1
        while notAdded:
            if abs(max_index) == len(yf):
                index_used = np.where(yf == np.sort(yf)[-1])[0]
                signal_f = xf[index_used][0]
                notAdded = False
            if xf[yf==np.sort(yf)[max_index]][0] < 0.5:
                max_index -= 1
            else:
                index_used = np.where(yf == np.sort(yf)[max_index])[0]
                signal_f = xf[index_used][0]
                notAdded = False

        if (signal_f >= threshold[0]) and (signal_f <=threshold[1]):
            y_test.append(1)
        else:
            y_test.append(0)  

    return y_train, y_val, y_test

def activity_filter(X_train: list, X_val: list, X_test: list, Y_train: list, Y_val: list, Y_test: list, Y_user_train: list, Y_user_val: list, Y_user_test: list, Y_train_activity: list, Y_val_activity: list, Y_test_activity: list):
    # Physiological data
    X_train_filt = [i[np.array(Y_train_activity)==0] for i in X_train]
    X_val_filt = [i[np.array(Y_val_activity)==0] for i in X_val]
    X_test_filt = [i[np.array(Y_test_activity)==0] for i in X_test]
    # Stress class
    Y_train_filt = Y_train[np.array(Y_train_activity)==0]
    Y_val_filt = Y_val[np.array(Y_val_activity)==0]
    Y_test_filt = Y_test[np.array(Y_test_activity)==0]
    # User class
    Y_user_train_filt = Y_user_train[np.array(Y_train_activity)==0]
    Y_user_val_filt = Y_user_val[np.array(Y_val_activity)==0]
    Y_user_test_filt = Y_user_test[np.array(Y_test_activity)==0]

    return X_train_filt, X_val_filt, X_test_filt, Y_train_filt, Y_val_filt, Y_test_filt, Y_user_train_filt, Y_user_val_filt, Y_user_test_filt

def get_activity_class(index_train: list, index_val: list, index_test: list, data_set: str, STD: bool=True, duration: int=300, path: str = "../Data/", before: int = 0, threshold: list = [40], SAMPLE_RATE: int = 32, seed: int = 123):
        '''
        Args:
                index_train (list): A list of of integers indicating where the training data is located.
                index_val (list): A list of of integers indicating where the validation data is located.
                index_test (list): A list of of integers indicating where the test data is located.
                data_set (str): The dataset we want to predict on.
                STD (boolean): If STD classification is using STD method else frequency method will be used. Default: ``True``.
                duration (int): Duration of signal in seconds. Default: 300sec.
                path (string): Path to the data. Default: "../Data/"
                before (int): The number of second before the tag are data is created from. Default: 0. 
                threshold (list): A list containing the threshold used for the classification, for frequency an lower and upper bound is given, for std only upper bound is given. Default:[].
                SAMPLE_RATE (int): The sampling rate for accelerometer. Default: 32

        Returns:
            Tuple (array, array, array): A binary activity classification for train, val, and test, where 0 is no physical activity and 1 is physical activity. 
        '''
        if STD:
                return activity_class_std(index_train = index_train, index_val = index_val, index_test = index_test, data_set = data_set, duration = duration, path = path, before = before, threshold = threshold, seed = seed)

        else:
                return activity_class_frequency(index_train = index_train, index_val = index_val, index_test = index_test, data_set = data_set, duration = duration,path = path, before = before, threshold = threshold, SAMPLE_RATE = SAMPLE_RATE, seed = seed)
