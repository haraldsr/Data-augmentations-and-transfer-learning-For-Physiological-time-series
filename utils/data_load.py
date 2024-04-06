from typing import List, Optional, Union
import numpy as np
import random
import pickle 
import pandas as pd
from scipy import signal
import scipy
import neurokit2 as nk
import sys

### Dataload, no standardization, but butterworth filter on BVP, EDA and TEMP + Sim
def data_load_multi(duration: int = 300, train_val: list = ['ADARP','DTU','ROAD','WESAD'], test: list = ['ADARP'], train_val_split: float = 0.2,  BW_filter: bool = False, path: str = '../Data/', seed: int = 123, include_SIM: bool = False, before: int = 0, SIM: str = ''):
        '''
        This is a helper function that loads the BVP, EDA, HR, and Temperature data (in that order) for multiple datasets. 

        Args:
                duration (int): The window/sequence length of a signal/instance. Default 300 (5min).
                train_val (list): The datasets to include in training and validation. Default ADARP, DTU, ROAD, and WESAD (All datasets).
                test (list): The datasets to include in test. Default: ADARP.
                train_val_split (float): The training and validation split. Default 0.2 (20% validation).
                BW_filter (bool): If the butterworth filter should be used. Default ``True``.
                path (str): Path to the data folder. Default: "../Data/"
                seed (int): Set the random seed for reproducibility. Default: 123.
                include_SIM (bool): If simulation data should be included. Default ``False``.
                before (int): The prediction lead time in seconds. Default 0.

        Returns:
                X, y ,and y subject. And simulation data if specified.
        '''
        
        # !!! Make sure to use new dataset as test !!!

        random.seed(seed)
        np.random.seed(seed)
        
        ### Train + Validation
        count = 0
        all_data_BVP = []
        all_data_EDA = []
        all_data_HR = []
        all_data_TEMP = []
        all_data_y = []
        if not before:
                before = ''
        for d in train_val:

                data_set = d
                folder = f"{data_set}_{duration}sec{'_bf'+str(before) if before else ''}/"
                file_str = f"{data_set}_{duration}sec{'_bf'+str(before) if before else ''}"
                with open(f'{path}{folder}{file_str}_BVP.pkl', 'rb') as f:
                        BVP = pickle.load(f)

                with open(f'{path}{folder}{file_str}_EDA.pkl', 'rb') as f:
                        EDA = pickle.load(f)

                with open(f'{path}{folder}{file_str}_HR.pkl', 'rb') as f:
                        HR = pickle.load(f)

                with open(f'{path}{folder}{file_str}_TEMP.pkl', 'rb') as f:
                        TEMP = pickle.load(f)
                
                num_y = sum([type(i)==str for i in BVP.columns]) # find number label columns
                mask_BVP = BVP.iloc[:,:-num_y].apply(lambda row: row.std()!=0, axis=1)
                mask_EDA = EDA.iloc[:,:-num_y].apply(lambda row: row.std()!=0, axis=1)
                mask_HR = HR.iloc[:,:-num_y].apply(lambda row: row.std()!=0, axis=1)
                mask_TEMP = TEMP.iloc[:,:-num_y].apply(lambda row: row.std()!=0, axis=1)
                per_stres_removed = sum(BVP["y_stress"][(~mask_BVP & ~mask_EDA & ~mask_HR & ~mask_TEMP)])/sum(BVP["y_stress"])
                if per_stres_removed:
                        print(f'Percent true labels filtered due to bad data: {per_stres_removed}')

                BVP = BVP[(mask_BVP & mask_EDA & mask_HR & mask_TEMP)].reset_index(drop=True)
                EDA = EDA[(mask_BVP & mask_EDA & mask_HR & mask_TEMP)].reset_index(drop=True)
                HR = HR[(mask_BVP & mask_EDA & mask_HR & mask_TEMP)].reset_index(drop=True)
                TEMP = TEMP[(mask_BVP & mask_EDA & mask_HR & mask_TEMP)].reset_index(drop=True)

                if count > 0:
                        BVP['subject'] = BVP['subject'] + max(all_data_BVP[count-1]['subject']) + 1

                if data_set == 'ADARP':
                        # Getting stress data                
                        amount_stress = len(BVP[BVP['y_stress']==1])
                        stress_idx = BVP.index[BVP['y_stress']==1]
                        BVP_stress = BVP.iloc[stress_idx,:]
                        EDA_stress = EDA.iloc[stress_idx,:]
                        HR_stress = HR.iloc[stress_idx,:]
                        TEMP_stress = TEMP.iloc[stress_idx,:]
                        # Getting no stress data
                        nostress_idx = BVP[BVP['y_stress']==0].sample(n = (amount_stress*2), 
                                                                      random_state=seed).index
                        BVP_nostress = BVP.iloc[nostress_idx,:]
                        EDA_nostress = EDA.iloc[nostress_idx,:]
                        HR_nostress = HR.iloc[nostress_idx,:]
                        TEMP_nostress = TEMP.iloc[nostress_idx,:]
                        # Combine
                        BVP = pd.concat([BVP_stress, BVP_nostress])
                        EDA = pd.concat([EDA_stress, EDA_nostress])
                        HR = pd.concat([HR_stress, HR_nostress])
                        TEMP = pd.concat([TEMP_stress, TEMP_nostress])

                all_data_BVP.append(BVP)
                all_data_EDA.append(EDA)
                all_data_HR.append(HR)
                all_data_TEMP.append(TEMP)
                all_data_y.append(BVP['y_stress'])

                count += 1

        ## Join data sets ##
        BVP = pd.concat(all_data_BVP).reset_index(drop=True)
        EDA = pd.concat(all_data_EDA).reset_index(drop=True)
        HR = pd.concat(all_data_HR).reset_index(drop=True)
        TEMP = pd.concat(all_data_TEMP).reset_index(drop=True)

        assert (len(BVP) == len(EDA) and len(HR)==len(TEMP) and len(BVP) == len(HR)) # Check if lengths are unequal

        num_y = sum([type(i)==str for i in BVP.columns]) # find number label columns

        if(BW_filter):
                ## Apply Butterworth filter
                # EDA and TEMP
                fs = 4  # Sampling frequency
                fc = 1  # Cut-off frequency of the filter
                w = fc / (fs / 2) # Normalize the frequency
                b, a = signal.butter(6, w, 'low',analog = False)
                EDA.iloc[:,:-num_y] = signal.filtfilt(b, a, EDA.iloc[:,0:-num_y])
                TEMP.iloc[:,:-num_y] = signal.filtfilt(b, a, TEMP.iloc[:,0:-num_y])
                # BVP
                fs = 64  # Sampling frequency
                fc_high = 2  # Cut-off frequency of the filter (high-pass)
                fc_low = 12 # Cut-off frequency for low pass
                #w = fc / (fs / 2) # Normalize the frequency
                sos = scipy.signal.butter(2, [fc_high, fc_low], btype="bandpass", output="sos", fs=fs)
                BVP.iloc[:,0:-num_y] = scipy.signal.sosfiltfilt(sos, BVP.iloc[:,0:-num_y])
                # b, a = signal.butter(2, w, 'low', analog = False)
                # BVP.iloc[:,:-num_y] = signal.filtfilt(b, a, BVP[:,:-num_y])
                # fc = 12  # Cut-off frequency of the filter
                # w = fc / (fs / 2) # Normalize the frequency
                # b, a = signal.butter(2, w, 'high', analog = False)
                # #BVP.iloc[:,:-num_y] = signal.filtfilt(b, a, BVP[:,:-num_y])
                # BVP.iloc[:,0:-num_y] = nk.signal_filter(BVP.iloc[:,0:-num_y], sampling_rate=64, highcut=12, lowcut=2, method="butterworth", order=2)


        # Get y values, stress and subject      
        y = BVP['y_stress'].values
        y_user = BVP['subject'].values # these values we create ourselves

        # Get unique labels
        unique_labels = np.unique(y_user)

        # Initialize empty DataFrames for training and testing
        index_train = []
        index_val = []

        # All index
        all_index = np.arange(0,y_user.shape[0])

        # Iterate over unique labels
        for label in unique_labels:
                # Get index with the current label
                label_idx = np.where(y_user==label)[0]
                np.random.shuffle(label_idx)

                # Split the rows into train and test sets
                train_group = label_idx[:1]
                val_group = label_idx[1:2]

                # Concatenate to the overall training and testing sets
                index_train.append(train_group[0])
                index_val.append(val_group[0])

        all_index = np.delete(all_index, [index_train, index_val])

        np.random.shuffle(all_index)

        amount_current_val = unique_labels.shape[0]
        amount_total_val = round(y_user.shape[0]*(1-train_val_split))
        amount_missing_val = amount_total_val-amount_current_val

        index_train = np.array(index_train)
        np.random.shuffle(index_train)
        index_val = np.array(index_val)
        np.random.shuffle(index_val)

        # Split index in train and validation
        index_train = np.append(index_train, all_index[:amount_missing_val])
        index_val = np.append(index_val, all_index[amount_missing_val:])

        ## splitting user data
        Y_user_train = y_user[index_train]
        Y_user_val = y_user[index_val]

        ## Check if all subjects are present in both train and val
        assert (set(Y_user_train) == set(Y_user_val))

        ## Creating datasets to return ##

        ## Train
        X_train = []
        X_train.append(BVP.iloc[index_train,:BVP.shape[1]-num_y].values)
        X_train.append(EDA.iloc[index_train,:EDA.shape[1]-num_y].values)
        X_train.append(HR.iloc[index_train,:HR.shape[1]-num_y].values)
        X_train.append(TEMP.iloc[index_train,:TEMP.shape[1]-num_y].values)
        # Y
        Y_train = y[index_train]
        if Y_train.mean() > 0.43 or Y_train.mean() < 0.23:
                print(f'\nWarning data is unbalanced. It is recommended that you use another seed to properly rebalance data.')
                print(f'Target balance is 0.33, but y data balance is: {Y_train.mean()}\n')
                #assert sum(Y_train)/len(Y_train) > 0.43 or sum(Y_train)/len(Y_train) < 0.23

        ## Validation
        X_val = []
        X_val.append(BVP.iloc[index_val,:BVP.shape[1]-num_y].values)
        X_val.append(EDA.iloc[index_val,:EDA.shape[1]-num_y].values)
        X_val.append(HR.iloc[index_val,:HR.shape[1]-num_y].values)
        X_val.append(TEMP.iloc[index_val,:TEMP.shape[1]-num_y].values)
        # Y
        Y_val = y[index_val]


        ### Simulated data 
        if(include_SIM):
                data_set = SIM[1:]
                #data_set = "SIMS"
                folder = f"{data_set}_{duration}sec/"
                file_str = f"{data_set}_{duration}sec"
                with open(f'{path}{folder}{file_str}_BVP.pkl', 'rb') as f:
                        BVP_sim = pickle.load(f)
                with open(f'{path}{folder}{file_str}_EDA.pkl', 'rb') as f:
                        EDA_sim = pickle.load(f)
                with open(f'{path}{folder}{file_str}_HR.pkl', 'rb') as f:
                        HR_sim = pickle.load(f)
                with open(f'{path}{folder}{file_str}_TEMP.pkl', 'rb') as f:
                        TEMP_sim = pickle.load(f)

                BVP_sim = BVP_sim.sample(frac = 1, random_state = seed)
                EDA_sim = EDA_sim.sample(frac = 1, random_state = seed)
                HR_sim = HR_sim.sample(frac = 1, random_state = seed)
                TEMP_sim = TEMP_sim.sample(frac = 1, random_state = seed)

                if(BW_filter):
                        ## Apply Butterworth filter
                        # EDA and TEMP
                        fs = 4  # Sampling frequency
                        fc = 1  # Cut-off frequency of the filter
                        w = fc / (fs / 2) # Normalize the frequency
                        b, a = signal.butter(6, w, 'low',analog = False)
                        EDA_sim.iloc[:,:] = signal.filtfilt(b, a, EDA_sim)
                        TEMP_sim.iloc[:,:] = signal.filtfilt(b, a, TEMP_sim)
                        # BVP
                        fs = 64  # Sampling frequency
                        # fc = 4  # Cut-off frequency of the filter
                        # w = fc / (fs / 2) # Normalize the frequency
                        # b, a = signal.butter(2, w, 'low',analog = False)
                        # BVP_sim.iloc[:,:] = signal.filtfilt(b, a, BVP_sim)
                        # b, a = signal.butter(12, w, 'high',analog = False)
                        # BVP_sim.iloc[:,:] = signal.filtfilt(b, a, BVP_sim)
                        fc_high = 2  # Cut-off frequency of the filter (high-pass)
                        fc_low = 12 # Cut-off frequency for low pass
                        #w = fc / (fs / 2) # Normalize the frequency
                        sos = scipy.signal.butter(2, [fc_high, fc_low], btype="bandpass", output="sos", fs=fs)
                        BVP.iloc[:,0:-num_y] = scipy.signal.sosfiltfilt(sos, BVP.iloc[:,0:-num_y])
                        #BVP.iloc[:,0:-num_y] = nk.signal_filter(BVP.iloc[:,0:-num_y], sampling_rate=64, highcut=12, lowcut=2, method="butterworth", order=2)
                        # BVP with nabian2018
                        # BVP_sim = nk.ppg_clean(BVP_sim, method="nabian2018", heart_rate = np.mean(HR_sim))
                
                X_train_sim = []
                X_val_sim = []
                
                amount_train_sim = round((1-train_val_split)*BVP_sim.shape[0])
                
                X_train_sim.append(np.append(X_train[0], BVP_sim.iloc[:amount_train_sim,], axis=0))
                np.random.shuffle(X_train_sim[0])
                X_train_sim.append(np.append(X_train[1], EDA_sim.iloc[:amount_train_sim,], axis=0))
                np.random.shuffle(X_train_sim[1])
                X_train_sim.append(np.append(X_train[2], HR_sim.iloc[:amount_train_sim,], axis=0))
                np.random.shuffle(X_train_sim[2])
                X_train_sim.append(np.append(X_train[3], TEMP_sim.iloc[:amount_train_sim,], axis=0))
                np.random.shuffle(X_train_sim[3])

                X_val_sim.append(np.append(X_val[0], BVP_sim.iloc[amount_train_sim:,], axis=0))
                np.random.shuffle(X_val_sim[0])
                X_val_sim.append(np.append(X_val[1], EDA_sim.iloc[amount_train_sim:,], axis=0))
                np.random.shuffle(X_val_sim[1])
                X_val_sim.append(np.append(X_val[2], HR_sim.iloc[amount_train_sim:,], axis=0))
                np.random.shuffle(X_val_sim[2])
                X_val_sim.append(np.append(X_val[3], TEMP_sim.iloc[amount_train_sim:,], axis=0))
                np.random.shuffle(X_val_sim[3])
        else:
                X_train_sim = None
                X_val_sim = None

        ### Test
        count_test = 0
        all_test_BVP = []
        all_test_EDA = []
        all_test_HR = []
        all_test_TEMP = []
        all_test_y = []
        for d in test:

                data_set = d
                folder = f"{data_set}_{duration}sec{'_bf'+str(before) if before else ''}/"
                file_str = f"{data_set}_{duration}sec{'_bf'+str(before) if before else ''}"
                with open(f'{path}{folder}{file_str}_BVP.pkl', 'rb') as f:
                        BVP = pickle.load(f).astype(float)

                with open(f'{path}{folder}{file_str}_EDA.pkl', 'rb') as f:
                        EDA = pickle.load(f).astype(float)

                with open(f'{path}{folder}{file_str}_HR.pkl', 'rb') as f:
                        HR = pickle.load(f).astype(float)

                with open(f'{path}{folder}{file_str}_TEMP.pkl', 'rb') as f:
                        TEMP = pickle.load(f).astype(float)
                
                if count_test > 0:
                        BVP['subject'] = BVP['subject'] + max(all_test_BVP[count_test-1]['subject']) + 1

                if data_set == 'ADARP':
                        # Getting stress data                
                        amount_stress = len(BVP[BVP['y_stress']==1])
                        stress_idx = BVP.index[BVP['y_stress']==1]
                        BVP_stress = BVP.iloc[stress_idx,:]
                        EDA_stress = EDA.iloc[stress_idx,:]
                        HR_stress = HR.iloc[stress_idx,:]
                        TEMP_stress = TEMP.iloc[stress_idx,:]
                        # Getting no stress data
                        nostress_idx = BVP[BVP['y_stress']==0].sample(n = (amount_stress*2), 
                                                                      random_state=seed).index
                        BVP_nostress = BVP.iloc[nostress_idx,:]
                        EDA_nostress = EDA.iloc[nostress_idx,:]
                        HR_nostress = HR.iloc[nostress_idx,:]
                        TEMP_nostress = TEMP.iloc[nostress_idx,:]
                        # Combine
                        BVP = pd.concat([BVP_stress, BVP_nostress])
                        EDA = pd.concat([EDA_stress, EDA_nostress])
                        HR = pd.concat([HR_stress, HR_nostress])
                        TEMP = pd.concat([TEMP_stress, TEMP_nostress])

                all_test_BVP.append(BVP)
                all_test_EDA.append(EDA)
                all_test_HR.append(HR)
                all_test_TEMP.append(TEMP)
                all_test_y.append(BVP['y_stress'])

                count_test += 1
        
        ## Join data sets ##
        BVP_test = pd.concat(all_test_BVP).reset_index(drop=True)
        EDA_test = pd.concat(all_test_EDA).reset_index(drop=True)
        HR_test = pd.concat(all_test_HR).reset_index(drop=True)
        TEMP_test = pd.concat(all_test_TEMP).reset_index(drop=True)

        assert (len(BVP_test) == len(EDA_test) and len(HR_test)==len(TEMP_test) and len(BVP_test) == len(HR_test)) #Check if lengths are unequal

        num_y_test = sum([type(i)==str for i in BVP_test.columns]) # find number label columns

        if(BW_filter):
                ## Apply Butterworth filter
                # EDA and TEMP
                fs = 4  # Sampling frequency
                fc = 1  # Cut-off frequency of the filter
                w = fc / (fs / 2) # Normalize the frequency
                b, a = signal.butter(6, w, 'low',analog = False)
                EDA_test.iloc[:,0:-num_y_test] = signal.filtfilt(b, a, EDA_test.iloc[:,0:-num_y_test])
                TEMP_test.iloc[:,0:-num_y_test] = signal.filtfilt(b, a, TEMP_test.iloc[:,0:-num_y_test])
                # BVP
                fs = 64  # Sampling frequency
                fc_high = 2  # Cut-off frequency of the filter (high-pass)
                fc_low = 12 # Cut-off frequency for low pass
                #w = fc / (fs / 2) # Normalize the frequency
                sos = scipy.signal.butter(2, [fc_high, fc_low], btype="bandpass", output="sos", fs=fs)
                BVP.iloc[:,0:-num_y] = scipy.signal.sosfiltfilt(sos, BVP.iloc[:,0:-num_y])
                # fc = 12  # Cut-off frequency of the filter
                # w = fc / (fs / 2) # Normalize the frequency
                # b, a = signal.butter(2, w, 'high', analog = False)
                # BVP.iloc[:,:-num_y] = signal.filtfilt(b, a, BVP[:,:-num_y])
                #BVP.iloc[:,0:-num_y] = nk.signal_filter(BVP.iloc[:,0:-num_y], sampling_rate=64, highcut=12, lowcut=2, method="butterworth", order=2)

        ## Get y values, stress and subject      
        Y_test = BVP_test['y_stress'].values
        Y_user_test = BVP_test['subject'].values # these values we create ourselves

        ## Creating datasets to return ##

        ## Train
        X_test = []
        X_test.append(BVP_test.iloc[:,:BVP_test.shape[1]-num_y_test].values)
        X_test.append(EDA_test.iloc[:,:EDA_test.shape[1]-num_y_test].values)
        X_test.append(HR_test.iloc[:,:HR_test.shape[1]-num_y_test].values)
        X_test.append(TEMP_test.iloc[:,:TEMP_test.shape[1]-num_y_test].values)


        return (X_train, X_val, X_test, Y_train, Y_val, Y_test, Y_user_train, Y_user_val, Y_user_test, X_train_sim, X_val_sim)

### Train, val and test on same dataset 
def data_load_single(duration: int = 300, data_set: str = 'ADARP', train_val_test_split: list = [0.8,0.1,0.1],  BW_filter = False, path='../Data/', seed = 123, before = 0):
        '''
        This is a helper function that loads the BVP, EDA, HR, and Temperature data (in that order) for one dataset. 
        Additionally, indexes which are used physical activity filtering are also returned. 

        Args:
                duration (int): The window/sequence length of a signal/instance. Default 300 (5min).
                data_set (str): The dataset to include. Default ADARP.
                train_val_split (float): The training, validation, and test split. Default [0.8,0.1,0.1] (80% training, 10% validation, and 10% testing).
                BW_filter (bool): If the butterworth filter should be used. Default ``True``.
                path (str): Path to the data folder. Default: "../Data/"
                seed (int): Set the random seed for reproducibility. Default: 123.
                before (int): The prediction lead time in seconds. Default 0.

        Returns:
                X, y, y subject, and indexes for data.
        '''

        assert sum(train_val_test_split) == 1.0
        random.seed(seed)
        np.random.seed(seed)

        all_data_BVP = []
        all_data_EDA = []
        all_data_HR = []
        all_data_TEMP = []
        all_data_y = []

        folder = f"{data_set}_{duration}sec{'_bf'+str(before) if before else ''}/"
        file_str = f"{data_set}_{duration}sec{'_bf'+str(before) if before else ''}"
        with open(f'{path}{folder}{file_str}_BVP.pkl', 'rb') as f:
                BVP = pickle.load(f)

        with open(f'{path}{folder}{file_str}_EDA.pkl', 'rb') as f:
                EDA = pickle.load(f)

        with open(f'{path}{folder}{file_str}_HR.pkl', 'rb') as f:
                HR = pickle.load(f)

        with open(f'{path}{folder}{file_str}_TEMP.pkl', 'rb') as f:
                TEMP = pickle.load(f)

        num_y = sum([type(i)==str for i in BVP.columns]) # find number label columns
        mask_BVP = BVP.iloc[:,:-num_y].apply(lambda row: row.std()!=0, axis=1)
        mask_EDA = EDA.iloc[:,:-num_y].apply(lambda row: row.std()!=0, axis=1)
        mask_HR = HR.iloc[:,:-num_y].apply(lambda row: row.std()!=0, axis=1)
        mask_TEMP = TEMP.iloc[:,:-num_y].apply(lambda row: row.std()!=0, axis=1)
        if np.sum(~mask_BVP & ~mask_EDA & ~mask_HR & ~mask_TEMP):
                print('Amount obs removed', np.sum(~mask_BVP & ~mask_EDA & ~mask_HR & ~mask_TEMP))
        per_stres_removed = sum(BVP["y_stress"][(~mask_BVP & ~mask_EDA & ~mask_HR & ~mask_TEMP)])/sum(BVP["y_stress"])
        if per_stres_removed:
                print(f'Percent true labels filtered due to bad data: {per_stres_removed}')
        BVP = BVP[(mask_BVP & mask_EDA & mask_HR & mask_TEMP)].reset_index(drop=True)
        EDA = EDA[(mask_BVP & mask_EDA & mask_HR & mask_TEMP)].reset_index(drop=True)
        HR = HR[(mask_BVP & mask_EDA & mask_HR & mask_TEMP)].reset_index(drop=True)
        TEMP = TEMP[(mask_BVP & mask_EDA & mask_HR & mask_TEMP)].reset_index(drop=True)

        if data_set in ['ADARP']:
                # Getting stress data                
                amount_stress = len(BVP[BVP['y_stress']==1])
                stress_idx = BVP.index[BVP['y_stress']==1]
                BVP_stress = BVP.iloc[stress_idx,:]
                EDA_stress = EDA.iloc[stress_idx,:]
                HR_stress = HR.iloc[stress_idx,:]
                TEMP_stress = TEMP.iloc[stress_idx,:]
                # Getting no stress data
                nostress_idx = BVP[BVP['y_stress']==0].sample(n = (amount_stress*2), 
                                                                random_state=seed).index
                BVP_nostress = BVP.iloc[nostress_idx,:]
                EDA_nostress = EDA.iloc[nostress_idx,:]
                HR_nostress = HR.iloc[nostress_idx,:]
                TEMP_nostress = TEMP.iloc[nostress_idx,:]
                # Combine
                BVP = pd.concat([BVP_stress, BVP_nostress])
                EDA = pd.concat([EDA_stress, EDA_nostress])
                HR = pd.concat([HR_stress, HR_nostress])
                TEMP = pd.concat([TEMP_stress, TEMP_nostress])

        all_data_BVP.append(BVP)
        all_data_EDA.append(EDA)
        all_data_HR.append(HR)
        all_data_TEMP.append(TEMP)
        all_data_y.append(BVP['y_stress'])


        ## Join data sets ##
        BVP = pd.concat(all_data_BVP).reset_index(drop=True)
        EDA = pd.concat(all_data_EDA).reset_index(drop=True)
        HR = pd.concat(all_data_HR).reset_index(drop=True)
        TEMP = pd.concat(all_data_TEMP).reset_index(drop=True)
        #y = pd.concat(all_data_y)

        assert (len(BVP) == len(EDA) and len(HR)==len(TEMP) and len(BVP) == len(HR)) # Check if lengths are unequal

        if(BW_filter):
                ## Apply Butterworth filter
                # EDA and TEMP
                fs = 4  # Sampling frequency
                fc = 1  # Cut-off frequency of the filter
                w = fc / (fs / 2) # Normalize the frequency
                b, a = signal.butter(6, w, 'low',analog = False)
                EDA.iloc[:,:-num_y] = signal.filtfilt(b, a, EDA.iloc[:,:-num_y])
                TEMP.iloc[:,:-num_y] = signal.filtfilt(b, a, TEMP.iloc[:,:-num_y])
                # BVP
                # fs = 64  # Sampling frequency
                # fc = 2  # Cut-off frequency of the filter
                # w = fc / (fs / 2) # Normalize the frequency
                # b, a = signal.butter(2, w, 'low', analog = False)
                # BVP.iloc[:,:-num_y] = signal.filtfilt(b, a, BVP.iloc[:,:-num_y])
                # fc = 12  # Cut-off frequency of the filter
                # w = fc / (fs / 2) # Normalize the frequency
                # b, a = signal.butter(2, w, 'high', analog = False)
                # BVP.iloc[:,:-num_y] = signal.filtfilt(b, a, BVP.iloc[:,:-num_y])
                fs = 64
                fc_high = 2  # Cut-off frequency of the filter (high-pass)
                fc_low = 12 # Cut-off frequency for low pass
                #w = fc / (fs / 2) # Normalize the frequency
                sos = scipy.signal.butter(2, [fc_high, fc_low], btype="bandpass", output="sos", fs=fs)
                BVP.iloc[:,0:-num_y] = scipy.signal.sosfiltfilt(sos, BVP.iloc[:,0:-num_y])
                #BVP.iloc[:,0:-num_y] = nk.signal_filter(BVP.iloc[:,0:-num_y], sampling_rate=64, highcut=12, lowcut=2, method="butterworth", order=2)
        # Get y values, stress and subject      
        y = BVP['y_stress'].values
        y_user = BVP['subject'].values # these values we create ourselves

        # Get unique labels
        unique_labels = np.unique(y_user)

        # Initialize empty DataFrames for training and testing
        index_train = []
        index_val = []
        index_test = []

        # All index
        all_index = np.arange(0,y_user.shape[0])

        # Iterate over unique labels
        for label in unique_labels:
                # Get index with the current label
                label_idx = np.where(y_user==label)[0]
                if len(label_idx) > 2:
                        np.random.shuffle(label_idx)

                        # Split the rows into train and test sets
                        train_group = label_idx[:1]
                        val_group = label_idx[1:2]
                        test_group = label_idx[2:3]

                        # Concatenate to the overall training and testing sets
                        index_train.append(train_group[0])
                        index_val.append(val_group[0])
                        index_test.append(test_group[0])

        all_index = np.delete(all_index, [index_train, index_val, index_test])

        np.random.shuffle(all_index)

        amount_current = unique_labels.shape[0]
        amount_total_train = round(y_user.shape[0]*train_val_test_split[0])
        amount_missing_train = amount_total_train-amount_current

        amount_total_val = round(y_user.shape[0]*train_val_test_split[1])
        amount_missing_val = amount_total_val-amount_current

        index_train = np.array(index_train)
        np.random.shuffle(index_train)
        index_val = np.array(index_val)
        np.random.shuffle(index_val)
        index_test = np.array(index_test)
        np.random.shuffle(index_test)

        # Split index in train and validation
        index_train = np.append(index_train, all_index[:amount_missing_train])
        index_val = np.append(index_val, all_index[amount_missing_train:(amount_missing_train+amount_missing_val)])
        index_test = np.append(index_test, all_index[(amount_missing_train+amount_missing_val):])

        assert set(index_train).difference(set(index_val)) == set(index_train)
        assert set(index_train).difference(set(index_test)) == set(index_train)
        assert set(index_val).difference(set(index_test)) == set(index_val)
        ## splitting user data
        Y_user_train = y_user[index_train]
        Y_user_val = y_user[index_val]
        Y_user_test = y_user[index_test]

        ## Check if all subjects are present in both train and val
        # assert (set(Y_user_train) == set(Y_user_val) == set(Y_user_test)) 

        # Ensure datasplit length ==  input data length
        assert len(Y_user_train)+len(Y_user_val)+len(Y_user_test) <= HR.shape[0]

        ## Creating datasets to return ##
        ## Train
        X_train = []
        X_train.append(BVP.iloc[index_train,:-num_y].values)
        X_train.append(EDA.iloc[index_train,:-num_y].values)
        X_train.append(HR.iloc[index_train,:-num_y].values)
        X_train.append(TEMP.iloc[index_train,:-num_y].values)
        # Y
        Y_train = y[index_train]

        if Y_train.mean() > 0.43 or Y_train.mean() < 0.23:
                print(f'\nWarning data is unbalanced. It is recommended that you use another seed to properly rebalance data.')
                print(f'Target balance is 0.33, but y data balance is: {(Y_train).mean()}\n')



        ## Validation
        X_val = []
        X_val.append(BVP.iloc[index_val,:-num_y].values)
        X_val.append(EDA.iloc[index_val,:-num_y].values)
        X_val.append(HR.iloc[index_val,:-num_y].values)
        X_val.append(TEMP.iloc[index_val,:-num_y].values)
        # Y
        Y_val = y[index_val]

        ## Test
        X_test = []
        X_test.append(BVP.iloc[index_test,:-num_y].values)
        X_test.append(EDA.iloc[index_test,:-num_y].values)
        X_test.append(HR.iloc[index_test,:-num_y].values)
        X_test.append(TEMP.iloc[index_test,:-num_y].values)
        # Y
        Y_test = y[index_test]

        return (X_train, X_val, X_test, Y_train, Y_val, Y_test, Y_user_train, Y_user_val, Y_user_test, index_train, index_val, index_test)

### For Random + Personalized Finetuning  
def data_load_personal(duration: int = 300, data_set: str = 'WA', train_val_split: float = 0.2, test_id: int = 1, days_to_finetune: int = 7, BW_filter: int = False, path: str ='../Data/', seed: int = 123, before: int = 0):
        

        random.seed(seed)
        np.random.seed(seed)

        all_data_BVP = []
        all_data_EDA = []
        all_data_HR = []
        all_data_TEMP = []
        all_data_y = []

        folder = f"{data_set}_{duration}sec{'_bf'+str(before) if before else ''}/"
        file_str = f"{data_set}_{duration}sec{'_bf'+str(before) if before else ''}"
        with open(f'{path}{folder}{file_str}_BVP.pkl', 'rb') as f:
                BVP = pickle.load(f)

        with open(f'{path}{folder}{file_str}_EDA.pkl', 'rb') as f:
                EDA = pickle.load(f)

        with open(f'{path}{folder}{file_str}_HR.pkl', 'rb') as f:
                HR = pickle.load(f).fillna(0)

        with open(f'{path}{folder}{file_str}_TEMP.pkl', 'rb') as f:
                TEMP = pickle.load(f)
        
        num_y = sum([type(i)==str for i in BVP.columns]) # find number label columns
        mask_BVP = BVP.iloc[:,:-num_y].apply(lambda row: row.std()!=0, axis=1)
        mask_EDA = EDA.iloc[:,:-num_y].apply(lambda row: row.std()!=0, axis=1)
        mask_HR = HR.iloc[:,:-num_y].apply(lambda row: row.std()!=0, axis=1)
        mask_TEMP = TEMP.iloc[:,:-num_y].apply(lambda row: row.std()!=0, axis=1)
        per_stres_removed = sum(BVP["y_stress"][(~mask_BVP & ~mask_EDA & ~mask_HR & ~mask_TEMP)])/sum(BVP["y_stress"])
        if per_stres_removed:
                print(f'Percent true labels filtered due to bad data: {per_stres_removed}')

        BVP = BVP[(mask_BVP & mask_EDA & mask_HR & mask_TEMP)].reset_index(drop=True)
        EDA = EDA[(mask_BVP & mask_EDA & mask_HR & mask_TEMP)].reset_index(drop=True)
        HR = HR[(mask_BVP & mask_EDA & mask_HR & mask_TEMP)].reset_index(drop=True)
        TEMP = TEMP[(mask_BVP & mask_EDA & mask_HR & mask_TEMP)].reset_index(drop=True)

        if data_set == 'ADARP':
                # Getting stress data                
                amount_stress = len(BVP[BVP['y_stress']==1])
                stress_idx = BVP.index[BVP['y_stress']==1]
                BVP_stress = BVP.iloc[stress_idx,:]
                EDA_stress = EDA.iloc[stress_idx,:]
                HR_stress = HR.iloc[stress_idx,:]
                TEMP_stress = TEMP.iloc[stress_idx,:]
                # Getting no stress data
                nostress_idx = BVP[BVP['y_stress']==0].sample(n = (amount_stress*2), 
                                                                random_state=seed).index
                BVP_nostress = BVP.iloc[nostress_idx,:]
                EDA_nostress = EDA.iloc[nostress_idx,:]
                HR_nostress = HR.iloc[nostress_idx,:]
                TEMP_nostress = TEMP.iloc[nostress_idx,:]
                # Combine
                BVP = pd.concat([BVP_stress, BVP_nostress])
                EDA = pd.concat([EDA_stress, EDA_nostress])
                HR = pd.concat([HR_stress, HR_nostress])
                TEMP = pd.concat([TEMP_stress, TEMP_nostress])

        all_data_BVP.append(BVP)
        all_data_EDA.append(EDA)
        all_data_HR.append(HR)
        all_data_TEMP.append(TEMP)
        all_data_y.append(BVP['y_stress'])


        ## Join data sets ##
        BVP = pd.concat(all_data_BVP).reset_index(drop=True)
        EDA = pd.concat(all_data_EDA).reset_index(drop=True)
        HR = pd.concat(all_data_HR).reset_index(drop=True)
        TEMP = pd.concat(all_data_TEMP).reset_index(drop=True)
        #y = pd.concat(all_data_y)

        assert (len(BVP) == len(EDA) and len(HR)==len(TEMP) and len(BVP) == len(HR)) # Check if lengths are unequal

        if(BW_filter):
                ## Apply Butterworth filter
                # EDA and TEMP
                fs = 4  # Sampling frequency
                fc = 1  # Cut-off frequency of the filter
                w = fc / (fs / 2) # Normalize the frequency
                b, a = signal.butter(6, w, 'low',analog = False)
                EDA.iloc[:,:-num_y] = signal.filtfilt(b, a, EDA.iloc[:,:-num_y])
                TEMP.iloc[:,:-num_y] = signal.filtfilt(b, a, TEMP.iloc[:,:-num_y])
                # BVP
                fs = 64  # Sampling frequency
                # fc = 2  # Cut-off frequency of the filter
                # w = fc / (fs / 2) # Normalize the frequency
                # b, a = signal.butter(2, w, 'low', analog = False)
                # BVP.iloc[:,:-num_y] = signal.filtfilt(b, a, BVP.iloc[:,:-num_y])
                # fc = 12  # Cut-off frequency of the filter
                # w = fc / (fs / 2) # Normalize the frequency
                # b, a = signal.butter(2, w, 'high', analog = False)
                # BVP.iloc[:,:-num_y] = signal.filtfilt(b, a, BVP.iloc[:,:-num_y])
                fc_high = 2  # Cut-off frequency of the filter (high-pass)
                fc_low = 12 # Cut-off frequency for low pass
                #w = fc / (fs / 2) # Normalize the frequency
                sos = scipy.signal.butter(2, [fc_high, fc_low], btype="bandpass", output="sos", fs=fs)
                BVP.iloc[:,0:-num_y] = scipy.signal.sosfiltfilt(sos, BVP.iloc[:,0:-num_y])
                # BVP.iloc[:,0:-num_y] = nk.signal_filter(BVP.iloc[:,0:-num_y], sampling_rate=64, highcut=12, lowcut=2, method="butterworth", order=2)
                # BVP with nabian2018
                # BVP.iloc[:,0:-num_y] = nk.ppg_clean(BVP.iloc[:,0:-num_y], method="nabian2018", heart_rate = np.mean(HR.iloc[:,0:-num_y]))


        # Get y values, stress and subject      
        y = BVP['y_stress'].values
        y_user = BVP['subject'].values # these values we create ourselves

        # Get unique labels
        unique_labels = np.unique(y_user[y_user!=test_id])

        # Initialize empty DataFrames for training and testing
        index_train_random = []
        index_val_random = []
        index_test_random = list(np.where(y_user==test_id)[0])

        # All index
        all_index = np.arange(0,y_user.shape[0])
        
        # Iterate over unique labels
        for label in unique_labels:
                # Get index with the current label
                label_idx = np.where(y_user==label)[0]
                np.random.shuffle(label_idx)

                # Split the rows into train and test sets
                train_group = label_idx[:1]
                val_group = label_idx[1:2]

                # Concatenate to the overall training and testing sets
                index_train_random.append(train_group[0])
                index_val_random.append(val_group[0])

        all_index = np.delete(all_index, index_train_random + index_val_random + index_test_random)

        np.random.shuffle(all_index)

        amount_current = unique_labels.shape[0]
        amount_total_train = round(y_user[y_user!=test_id].shape[0]*(1-train_val_split))
        amount_missing_train = amount_total_train-amount_current

        index_train_random = np.array(index_train_random)
        np.random.shuffle(index_train_random)
        index_val_random = np.array(index_val_random)
        np.random.shuffle(index_val_random)

        # Split index in train and validation
        index_train_random = np.append(index_train_random, all_index[:amount_missing_train])
        index_val_random = np.append(index_val_random, all_index[amount_missing_train:])

        ## splitting user data
        Y_user_train_random = y_user[index_train_random]
        Y_user_val_random = y_user[index_val_random]
        Y_user_test_random = y_user[index_test_random]

        ## Check if all subjects are present in both train and val
        assert (set(Y_user_train_random) == set(Y_user_val_random))
        # Check if only desired subject are present in test
        assert (set([test_id]) == set(Y_user_test_random)) 
        
        ## Creating datasets to return ##

        ## Train
        X_train_random = []
        X_train_random.append(BVP.iloc[index_train_random,:-num_y].values)
        X_train_random.append(EDA.iloc[index_train_random,:-num_y].values)
        X_train_random.append(HR.iloc[index_train_random,:-num_y].values)
        X_train_random.append(TEMP.iloc[index_train_random,:-num_y].values)
        # Y
        Y_train_random = y[index_train_random]

        if Y_train_random.mean() > 0.43 or Y_train_random.mean() < 0.23:
                print(f'\nWarning data is unbalanced. It is recommended that you use another seed to properly rebalance data.')
                print(f'Target balance is 0.33, but y data balance is: {Y_train_random.mean()}\n')



        ## Validation
        X_val_random = []
        X_val_random.append(BVP.iloc[index_val_random,:-num_y].values)
        X_val_random.append(EDA.iloc[index_val_random,:-num_y].values)
        X_val_random.append(HR.iloc[index_val_random,:-num_y].values)
        X_val_random.append(TEMP.iloc[index_val_random,:-num_y].values)
        # Y
        Y_val_random = y[index_val_random]

        ## Test
        X_test_random = []
        X_test_random.append(BVP.iloc[index_test_random,:-num_y].values)
        X_test_random.append(EDA.iloc[index_test_random,:-num_y].values)
        X_test_random.append(HR.iloc[index_test_random,:-num_y].values)
        X_test_random.append(TEMP.iloc[index_test_random,:-num_y].values)
        # Y
        Y_test_random = y[index_test_random]



        ########## PERSONALIZED FINE-TUNE ##########
        # Moving on with: X_test_random, Y_test_random, Y_user_test_random, index_test_random
        days = BVP['days'].values[index_test_random]

        # Get unique labels
        unique_labels = np.unique(days[days<days_to_finetune])

        index_train_personal = []
        index_val_personal = []
        index_test_personal = list(np.where(days>=days_to_finetune)[0])

        # All index
        all_index = np.arange(0,days.shape[0])

        # Iterate over unique labels
        for label in unique_labels:
                # Get index with the current label
                label_idx = np.where(days==label)[0]
                if(len(label_idx)>1):
                        np.random.shuffle(label_idx)

                        # Split the rows into train and test sets
                        train_group = label_idx[:1]
                        val_group = label_idx[1:2]

                        # Concatenate to the overall training and testing sets
                        index_train_personal.append(train_group[0])
                        index_val_personal.append(val_group[0])

        all_index = np.delete(all_index, index_train_personal + index_val_personal + index_test_personal)

        np.random.shuffle(all_index)

        amount_current = len(index_train_personal)
        amount_total_train = round(days[days<days_to_finetune].shape[0]*(1-train_val_split))
        amount_missing_train = amount_total_train-amount_current

        index_train_personal = np.array(index_train_personal)
        np.random.shuffle(index_train_personal)
        index_val_personal = np.array(index_val_personal)
        np.random.shuffle(index_val_personal)

        # Split index in train and validation
        index_train_personal = np.append(index_train_personal, all_index[:amount_missing_train])
        index_val_personal = np.append(index_val_personal, all_index[amount_missing_train:])

        ## splitting user data
        Y_user_train_personal = Y_user_test_random[index_train_personal]
        Y_user_val_personal = Y_user_test_random[index_val_personal]
        Y_user_test_personal = Y_user_test_random[index_test_personal]

        ## Check if only desired subject is present in train, val, and test
        #assert (set(Y_user_train_personal) == set(Y_user_val_personal) == set([test_id]) == set(Y_user_test_personal))
        if not (set(Y_user_train_personal) == set(Y_user_val_personal) == set([test_id]) == set(Y_user_test_personal)):
                sys.exit(0)

        ## Creating datasets to return ##

        ## Train
        X_train_personal = []
        X_train_personal.append(X_test_random[0][index_train_personal])
        X_train_personal.append(X_test_random[1][index_train_personal])
        X_train_personal.append(X_test_random[2][index_train_personal])
        X_train_personal.append(X_test_random[3][index_train_personal])
        # Y
        Y_train_personal = Y_test_random[index_train_personal]

        if Y_train_personal.mean() > 0.43 or Y_train_personal.mean() < 0.23:
                print(f'\nWarning data is unbalanced. It is recommended that you use another seed to properly rebalance data.')
                print(f'Target balance is 0.33, but y data balance is: {Y_train_personal.mean()}\n')

        ## Validation
        X_val_personal = []
        X_val_personal.append(X_test_random[0][index_val_personal])
        X_val_personal.append(X_test_random[1][index_val_personal])
        X_val_personal.append(X_test_random[2][index_val_personal])
        X_val_personal.append(X_test_random[3][index_val_personal])
        # Y
        Y_val_personal = Y_test_random[index_val_personal]

        ## Test
        X_test_personal = []
        X_test_personal.append(X_test_random[0][index_test_personal])
        X_test_personal.append(X_test_random[1][index_test_personal])
        X_test_personal.append(X_test_random[2][index_test_personal])
        X_test_personal.append(X_test_random[3][index_test_personal])
        # Y
        Y_test_personal = Y_test_random[index_test_personal]

        ### Save data info to textfile 
        txt_file = [
                f"ID: {test_id}\n",
                f"Number of observations in train random fine-tune: {len(X_train_random[0])} ({(len(X_train_random[0])/(len(X_train_random[0])+len(X_val_random[0])+len(X_test_random[0]))*100):.2}%) - Percentage stress: {(Y_train_random.mean()*100):.2}\n",
                f"Number of observations in val random fine-tune: {len(X_val_random[0])} ({(len(X_val_random[0])/(len(X_train_random[0])+len(X_val_random[0])+len(X_test_random[0]))*100):.2}%) - Percentage stress: {(Y_val_random.mean()*100):.2}\n",
                f"Number of observations in test random fine-tune: {len(X_test_random[0])} ({(len(X_test_random[0])/(len(X_train_random[0])+len(X_val_random[0])+len(X_test_random[0]))*100):.2}%) - Percentage stress: {(Y_test_random.mean()*100):.2}\n",
                f"Number of observations in train personal fine-tune: {len(X_train_personal[0])} ({(len(X_train_personal[0])/(len(X_train_personal[0])+len(X_val_personal[0])+len(X_test_personal[0]))*100):.2}%) - Percentage stress: {(Y_train_personal.mean()*100)}\n",
                f"Number of observations in val personal fine-tune: {len(X_val_personal[0])} ({(len(X_val_personal[0])/(len(X_train_personal[0])+len(X_val_personal[0])+len(X_test_personal[0]))*100):.2}%) - Percentage stress: {(Y_val_personal.mean()*100):.2}\n",
                f"Number of observations in test personal fine-tune: {len(X_test_personal[0])} ({(len(X_test_personal[0])/(len(X_train_personal[0])+len(X_val_personal[0])+len(X_test_personal[0]))*100):.2}%) - Percentage stress: {(Y_test_personal.mean()*100):.2}"
        ]
        with open(f'data_info\personal_data_info_{data_set}_duration{duration}_bf{before}_id{test_id}_seed{seed}.txt', 'w') as f:
                f.writelines(txt_file)         

        return (X_train_random, X_val_random, X_test_random, 
                Y_train_random, Y_val_random, Y_test_random, 
                Y_user_train_random, Y_user_val_random, Y_user_test_random, 
                index_train_random, index_val_random, index_test_random,
                X_train_personal, X_val_personal, X_test_personal, 
                Y_train_personal, Y_val_personal, Y_test_personal, 
                Y_user_train_personal, Y_user_val_personal, Y_user_test_personal, 
                index_train_personal, index_val_personal, index_test_personal)

### Train, val and test on same dataset - Include sim
def data_load_single_SIM(duration: int = 300, data_set: str = 'ADARP', train_val_test_split: list = [0.8,0.1,0.1],  BW_filter = False, path='../Data/', seed = 123, before = 0, SIM: str = 'SIMC'):
        '''
        This is a helper function that loads the BVP, EDA, HR, and Temperature data (in that order) for one dataset. 
        Additionally, indexes which are used physical activity filtering are also returned. 

        Args:
                duration (int): The window/sequence length of a signal/instance. Default 300 (5min).
                data_set (str): The dataset to include. Default ADARP.
                train_val_split (float): The training, validation, and test split. Default [0.8,0.1,0.1] (80% training, 10% validation, and 10% testing).
                BW_filter (bool): If the butterworth filter should be used. Default ``True``.
                path (str): Path to the data folder. Default: "../Data/"
                seed (int): Set the random seed for reproducibility. Default: 123.
                before (int): The prediction lead time in seconds. Default 0.

        Returns:
                X, y, y subject, and indexes for data.
        '''

        assert sum(train_val_test_split) == 1.0
        random.seed(seed)
        np.random.seed(seed)

        all_data_BVP = []
        all_data_EDA = []
        all_data_HR = []
        all_data_TEMP = []
        all_data_y = []

        folder = f"{data_set}_{duration}sec{'_bf'+str(before) if before else ''}/"
        file_str = f"{data_set}_{duration}sec{'_bf'+str(before) if before else ''}"
        with open(f'{path}{folder}{file_str}_BVP.pkl', 'rb') as f:
                BVP = pickle.load(f)

        with open(f'{path}{folder}{file_str}_EDA.pkl', 'rb') as f:
                EDA = pickle.load(f)

        with open(f'{path}{folder}{file_str}_HR.pkl', 'rb') as f:
                HR = pickle.load(f)

        with open(f'{path}{folder}{file_str}_TEMP.pkl', 'rb') as f:
                TEMP = pickle.load(f)

        num_y = sum([type(i)==str for i in BVP.columns]) # find number label columns
        mask_BVP = BVP.iloc[:,:-num_y].apply(lambda row: row.std()!=0, axis=1)
        mask_EDA = EDA.iloc[:,:-num_y].apply(lambda row: row.std()!=0, axis=1)
        mask_HR = HR.iloc[:,:-num_y].apply(lambda row: row.std()!=0, axis=1)
        mask_TEMP = TEMP.iloc[:,:-num_y].apply(lambda row: row.std()!=0, axis=1)
        if np.sum(~mask_BVP & ~mask_EDA & ~mask_HR & ~mask_TEMP):
                print('Amount obs removed', np.sum(~mask_BVP & ~mask_EDA & ~mask_HR & ~mask_TEMP))
        per_stres_removed = sum(BVP["y_stress"][(~mask_BVP & ~mask_EDA & ~mask_HR & ~mask_TEMP)])/sum(BVP["y_stress"])
        if per_stres_removed:
                print(f'Percent true labels filtered due to bad data: {per_stres_removed}')
        BVP = BVP[(mask_BVP & mask_EDA & mask_HR & mask_TEMP)].reset_index(drop=True)
        EDA = EDA[(mask_BVP & mask_EDA & mask_HR & mask_TEMP)].reset_index(drop=True)
        HR = HR[(mask_BVP & mask_EDA & mask_HR & mask_TEMP)].reset_index(drop=True)
        TEMP = TEMP[(mask_BVP & mask_EDA & mask_HR & mask_TEMP)].reset_index(drop=True)

        if data_set in ['ADARP']:
                # Getting stress data                
                amount_stress = len(BVP[BVP['y_stress']==1])
                stress_idx = BVP.index[BVP['y_stress']==1]
                BVP_stress = BVP.iloc[stress_idx,:]
                EDA_stress = EDA.iloc[stress_idx,:]
                HR_stress = HR.iloc[stress_idx,:]
                TEMP_stress = TEMP.iloc[stress_idx,:]
                # Getting no stress data
                nostress_idx = BVP[BVP['y_stress']==0].sample(n = (amount_stress*2), 
                                                                random_state=seed).index
                BVP_nostress = BVP.iloc[nostress_idx,:]
                EDA_nostress = EDA.iloc[nostress_idx,:]
                HR_nostress = HR.iloc[nostress_idx,:]
                TEMP_nostress = TEMP.iloc[nostress_idx,:]
                # Combine
                BVP = pd.concat([BVP_stress, BVP_nostress])
                EDA = pd.concat([EDA_stress, EDA_nostress])
                HR = pd.concat([HR_stress, HR_nostress])
                TEMP = pd.concat([TEMP_stress, TEMP_nostress])

        all_data_BVP.append(BVP)
        all_data_EDA.append(EDA)
        all_data_HR.append(HR)
        all_data_TEMP.append(TEMP)
        all_data_y.append(BVP['y_stress'])


        ## Join data sets ##
        BVP = pd.concat(all_data_BVP).reset_index(drop=True)
        EDA = pd.concat(all_data_EDA).reset_index(drop=True)
        HR = pd.concat(all_data_HR).reset_index(drop=True)
        TEMP = pd.concat(all_data_TEMP).reset_index(drop=True)
        #y = pd.concat(all_data_y)

        assert (len(BVP) == len(EDA) and len(HR)==len(TEMP) and len(BVP) == len(HR)) # Check if lengths are unequal

        if(BW_filter):
                ## Apply Butterworth filter
                # EDA and TEMP
                fs = 4  # Sampling frequency
                fc = 1  # Cut-off frequency of the filter
                w = fc / (fs / 2) # Normalize the frequency
                b, a = signal.butter(6, w, 'low',analog = False)
                EDA.iloc[:,:-num_y] = signal.filtfilt(b, a, EDA.iloc[:,:-num_y])
                TEMP.iloc[:,:-num_y] = signal.filtfilt(b, a, TEMP.iloc[:,:-num_y])
                # BVP
                # fs = 64  # Sampling frequency
                # fc = 2  # Cut-off frequency of the filter
                # w = fc / (fs / 2) # Normalize the frequency
                # b, a = signal.butter(2, w, 'low', analog = False)
                # BVP.iloc[:,:-num_y] = signal.filtfilt(b, a, BVP.iloc[:,:-num_y])
                # fc = 12  # Cut-off frequency of the filter
                # w = fc / (fs / 2) # Normalize the frequency
                # b, a = signal.butter(2, w, 'high', analog = False)
                # BVP.iloc[:,:-num_y] = signal.filtfilt(b, a, BVP.iloc[:,:-num_y])
                fs = 64
                fc_high = 2  # Cut-off frequency of the filter (high-pass)
                fc_low = 12 # Cut-off frequency for low pass
                #w = fc / (fs / 2) # Normalize the frequency
                sos = scipy.signal.butter(2, [fc_high, fc_low], btype="bandpass", output="sos", fs=fs)
                BVP.iloc[:,0:-num_y] = scipy.signal.sosfiltfilt(sos, BVP.iloc[:,0:-num_y])
                #BVP.iloc[:,0:-num_y] = nk.signal_filter(BVP.iloc[:,0:-num_y], sampling_rate=64, highcut=12, lowcut=2, method="butterworth", order=2)
        # Get y values, stress and subject      
        y = BVP['y_stress'].values
        y_user = BVP['subject'].values # these values we create ourselves

        # Get unique labels
        unique_labels = np.unique(y_user)

        # Initialize empty DataFrames for training and testing
        index_train = []
        index_val = []
        index_test = []

        # All index
        all_index = np.arange(0,y_user.shape[0])

        # Iterate over unique labels
        for label in unique_labels:
                # Get index with the current label
                label_idx = np.where(y_user==label)[0]
                if len(label_idx) > 2:
                        np.random.shuffle(label_idx)

                        # Split the rows into train and test sets
                        train_group = label_idx[:1]
                        val_group = label_idx[1:2]
                        test_group = label_idx[2:3]

                        # Concatenate to the overall training and testing sets
                        index_train.append(train_group[0])
                        index_val.append(val_group[0])
                        index_test.append(test_group[0])

        all_index = np.delete(all_index, [index_train, index_val, index_test])

        np.random.shuffle(all_index)

        amount_current = unique_labels.shape[0]
        amount_total_train = round(y_user.shape[0]*train_val_test_split[0])
        amount_missing_train = amount_total_train-amount_current

        amount_total_val = round(y_user.shape[0]*train_val_test_split[1])
        amount_missing_val = amount_total_val-amount_current

        index_train = np.array(index_train)
        np.random.shuffle(index_train)
        index_val = np.array(index_val)
        np.random.shuffle(index_val)
        index_test = np.array(index_test)
        np.random.shuffle(index_test)

        # Split index in train and validation
        index_train = np.append(index_train, all_index[:amount_missing_train])
        index_val = np.append(index_val, all_index[amount_missing_train:(amount_missing_train+amount_missing_val)])
        index_test = np.append(index_test, all_index[(amount_missing_train+amount_missing_val):])

        assert set(index_train).difference(set(index_val)) == set(index_train)
        assert set(index_train).difference(set(index_test)) == set(index_train)
        assert set(index_val).difference(set(index_test)) == set(index_val)
        ## splitting user data
        Y_user_train = y_user[index_train]
        Y_user_val = y_user[index_val]
        Y_user_test = y_user[index_test]

        ## Check if all subjects are present in both train and val
        # assert (set(Y_user_train) == set(Y_user_val) == set(Y_user_test)) 

        # Ensure datasplit length ==  input data length
        assert len(Y_user_train)+len(Y_user_val)+len(Y_user_test) <= HR.shape[0]

        ## Creating datasets to return ##
        ## Train
        X_train = []
        X_train.append(BVP.iloc[index_train,:-num_y].values)
        X_train.append(EDA.iloc[index_train,:-num_y].values)
        X_train.append(HR.iloc[index_train,:-num_y].values)
        X_train.append(TEMP.iloc[index_train,:-num_y].values)
        # Y
        Y_train = y[index_train]

        if Y_train.mean() > 0.43 or Y_train.mean() < 0.23:
                print(f'\nWarning data is unbalanced. It is recommended that you use another seed to properly rebalance data.')
                print(f'Target balance is 0.33, but y data balance is: {(Y_train).mean()}\n')



        ## Validation
        X_val = []
        X_val.append(BVP.iloc[index_val,:-num_y].values)
        X_val.append(EDA.iloc[index_val,:-num_y].values)
        X_val.append(HR.iloc[index_val,:-num_y].values)
        X_val.append(TEMP.iloc[index_val,:-num_y].values)
        # Y
        Y_val = y[index_val]

        ## Test
        X_test = []
        X_test.append(BVP.iloc[index_test,:-num_y].values)
        X_test.append(EDA.iloc[index_test,:-num_y].values)
        X_test.append(HR.iloc[index_test,:-num_y].values)
        X_test.append(TEMP.iloc[index_test,:-num_y].values)
        # Y
        Y_test = y[index_test]

        ### Simulated data 
        data_set = SIM[1:]
        folder = f"{data_set}_{duration}sec/"
        file_str = f"{data_set}_{duration}sec"
        with open(f'{path}{folder}{file_str}_BVP.pkl', 'rb') as f:
                BVP_sim = pickle.load(f)
        with open(f'{path}{folder}{file_str}_EDA.pkl', 'rb') as f:
                EDA_sim = pickle.load(f)
        with open(f'{path}{folder}{file_str}_HR.pkl', 'rb') as f:
                HR_sim = pickle.load(f)
        with open(f'{path}{folder}{file_str}_TEMP.pkl', 'rb') as f:
                TEMP_sim = pickle.load(f)

        BVP_sim = BVP_sim.sample(frac = 1, random_state = seed)
        EDA_sim = EDA_sim.sample(frac = 1, random_state = seed)
        HR_sim = HR_sim.sample(frac = 1, random_state = seed)
        TEMP_sim = TEMP_sim.sample(frac = 1, random_state = seed)

        if(BW_filter):
                ## Apply Butterworth filter
                # EDA and TEMP
                fs = 4  # Sampling frequency
                fc = 1  # Cut-off frequency of the filter
                w = fc / (fs / 2) # Normalize the frequency
                b, a = signal.butter(6, w, 'low',analog = False)
                EDA_sim.iloc[:,:] = signal.filtfilt(b, a, EDA_sim)
                TEMP_sim.iloc[:,:] = signal.filtfilt(b, a, TEMP_sim)
                # BVP
                fs = 64  # Sampling frequency
                fc = 4  # Cut-off frequency of the filter
                w = fc / (fs / 2) # Normalize the frequency
                b, a = signal.butter(6, w, 'low',analog = False)
                BVP_sim.iloc[:,:] = signal.filtfilt(b, a, BVP_sim)
                # BVP with nabian2018
                # BVP_sim = nk.ppg_clean(BVP_sim, method="nabian2018", heart_rate = np.mean(HR_sim))
        
        X_train_sim = []
        X_val_sim = []
        
        amount_train_sim = round((1-train_val_test_split[1])*BVP_sim.shape[0])
        
        X_train_sim.append(np.append(X_train[0], BVP_sim.iloc[:amount_train_sim,], axis=0))
        np.random.shuffle(X_train_sim[0])
        X_train_sim.append(np.append(X_train[1], EDA_sim.iloc[:amount_train_sim,], axis=0))
        np.random.shuffle(X_train_sim[1])
        X_train_sim.append(np.append(X_train[2], HR_sim.iloc[:amount_train_sim,], axis=0))
        np.random.shuffle(X_train_sim[2])
        X_train_sim.append(np.append(X_train[3], TEMP_sim.iloc[:amount_train_sim,], axis=0))
        np.random.shuffle(X_train_sim[3])

        X_val_sim.append(np.append(X_val[0], BVP_sim.iloc[amount_train_sim:,], axis=0))
        np.random.shuffle(X_val_sim[0])
        X_val_sim.append(np.append(X_val[1], EDA_sim.iloc[amount_train_sim:,], axis=0))
        np.random.shuffle(X_val_sim[1])
        X_val_sim.append(np.append(X_val[2], HR_sim.iloc[amount_train_sim:,], axis=0))
        np.random.shuffle(X_val_sim[2])
        X_val_sim.append(np.append(X_val[3], TEMP_sim.iloc[amount_train_sim:,], axis=0))
        np.random.shuffle(X_val_sim[3])
        

        return (X_train, X_val, X_test, Y_train, Y_val, Y_test, Y_user_train, Y_user_val, Y_user_test, index_train, index_val, index_test, X_train_sim, X_val_sim)

def get_data_load(test: List[str], data_load: str = 'multi', train_val: Optional[List[str]] = None, duration: int = 300, split: Union[float, List[float]] = 0.2, BW_filter: bool = True, seed: int = 123, path: str = '../Data/', include_SIM: bool = False, before: int = 0, test_id: int=1, SIM: str=''):
    '''
    Args:
        test: List with names of the datasets to include in the test data
        multi: If multiple datasets should be included. Default: ``True``.
        train_val: List with names of the datasets to include in the training and validation data. Default: ``None``.
        duration: Duration of signal in seconds. Default: 300sec.
        split: Percentage of data to include in validation if double (multi=``True``) or train, val, and test splits of data if list (multi=``False``). Default: 0.2.
        BW_filter: If data should be filtered with the butterworth filter. Default: ``True``.
        seed: Set the random seed. Default: 123.
        path: Path to the data. Default: "../Data/".
        include_SIM: If simulated data should be included. Default ``False``.

        Returns: 
                Blood Volume Pulse, Electrodermal Activity, Heart Rate, Temperature, stress, subject data for specified datasets.
    '''
    if data_load == 'multi':
        return data_load_multi(duration=duration, train_val=train_val, test=test, train_val_split=split, BW_filter=BW_filter, path=path, seed=seed, include_SIM=include_SIM, before=before, SIM=SIM)
    elif data_load == 'single':
        return data_load_single(duration=duration, data_set=test[0], train_val_test_split=split, BW_filter=BW_filter, path=path, seed=seed, before=before)
    elif data_load == 'single_SIM':
        return data_load_single_SIM(duration=duration, data_set=test[0], train_val_test_split=split, BW_filter=BW_filter, path=path, seed=seed, before=before, SIM=SIM)
    elif data_load == 'personal':
        return data_load_personal(duration = duration, data_set = test[0], train_val_split = split, test_id = test_id, days_to_finetune = 7, BW_filter = BW_filter, path = path, seed = seed, before = before)