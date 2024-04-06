import numpy as np
# Standardize by each observation
def standardize_obs(data):
    if type(data) == list:
        for i, X in enumerate(data):
            data[i] = (X-X.mean(axis=1)[:,None])/X.std(axis=1)[:, None]
            return data
    else:
        return (data-data.mean(axis=1)[:,None])/data.std(axis=1)[:, None]

#standardize by subject
def standardize_subject(X_data: list, y_subject, train: bool = False, subject_stat_dict = None):
    if train:
        subject_stat_dict = {}
        unique_subject = set(y_subject)
        for sub in unique_subject:
            RowWithSubject = np.where(y_subject == sub)[0]
            subjectMean =  [np.mean(i[RowWithSubject]) for i in X_data]
            subjectSTD = [np.std(i[RowWithSubject]) for i in X_data]
            subject_stat_dict[sub]= {'mean': subjectMean,
                                     'std': subjectSTD}            
            for r in range(len(X_data)):
                X_data[r][RowWithSubject] = (X_data[r][RowWithSubject]-subjectMean[r])/subjectSTD[r]
        return X_data, subject_stat_dict  
    
    elif subject_stat_dict:
        unique_subject = set(y_subject)
        for sub in unique_subject:
            RowWithSubject = np.where(y_subject == sub)[0]
            for r in range(len(X_data)):
                X_data[r][RowWithSubject] = (X_data[r][RowWithSubject]-subject_stat_dict[sub]['mean'][r])/subject_stat_dict[sub]['std'][r]
        return X_data
    else: 
        raise Exception("We do not have the correct input to do standardization ")

# function to call each standardization case
def get_standardize(X_train: list, X_val: list, X_test: list, method: str='obs', dim: str='all', y_subject_train: np.array=None, y_subject_val: np.array=None, y_subject_test: np.array=None) ->  tuple[list, list, list]:
    '''
    Standardizes the data by the given method and dimensions.

    Args:
        X_train (list): List of training data for the different biosignals.
        X_val (list): List of validation data for the different biosignals.
        X_test (list): List of test data for the different biosignals.
        method (string): If ``obs`` - standardization by observation. If ``sub`` - standardization by subject. Default: "obs".
        dim (str/list): If ``all`` - all dimension included in standardization. If list is specified, standardization is only done for given dimension. Default: "all".
        y_subject_train (np.array): The subject IDs for training. Only used for standardization by subject. Default: None.
        y_subject_val (np.array): . The subject IDs for validation. Only used for standardization by subject. Default: None.
        y_subject_test (np.array): . The subject IDs for test. Only used for standardization by subject. Default: None.

    Returns:
        tuple (list, list, list): A tuple with the standardized train, val, and test data.
    '''
    if method == 'obs':
        if dim == 'all':
            X_train = standardize_obs(X_train)
            X_val = standardize_obs(X_val)
            X_test = standardize_obs(X_test)
        else:
            if dim:
                for i in dim:
                    X_train[i] = standardize_obs(X_train[i])
                    X_val[i] = standardize_obs(X_val[i])
                    X_test[i] = standardize_obs(X_test[i])
        return X_train, X_val, X_test
    elif method == 'sub':
        X_train, subject_stat_dict = standardize_subject(X_train, y_subject_train, train = True)
        X_val = standardize_subject(X_val, y_subject_val, subject_stat_dict = subject_stat_dict)
        X_test = standardize_subject(X_test, y_subject_test, subject_stat_dict = subject_stat_dict)
        return X_train, X_val, X_test
    else:
        raise ValueError(f'Standardization method "{method}" not implemented yet.')

        
