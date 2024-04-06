import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
import argparse
import ast
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import sys

#Scripts created
from models.auto_class import getModel
from utils.standardize import get_standardize
from utils.data_load import get_data_load
from utils.train import train_autoencoder, train_classification, train_subject
from utils.predict import predict
from utils.DataUtils import get_freq, convert_X_tensor, get_freq_single, convert_X_tensor_single
from utils.activity_class import get_activity_class, activity_filter
from utils.config import model_config, auto_config, classi_config, sub_diff_config
from utils.DataUtils import E4Data_freq, E4Data, E4Data_freq_stratify


def main(args):
    # New folder name (if specified)
    if args.folder_arg:
        args.folder_arg = '_' + args.folder_arg

    # To add more details to name of model (if specified)
    if args.additional_model_details:
        args.additional_model_details = '_' + args.additional_model_details

    # Number of hidden nodes for each frequency
    nodes_per_freq = str(ast.literal_eval(args.n1_n2))
    nodes_per_freq = list(map(int, nodes_per_freq.replace("[","").replace("]","").split(', ')))

    # Kernel sizes
    kernel_size = ast.literal_eval(args.kernel_size)

    # To load arguments as lists
    train_val_datasets = args.train_val.strip('[]').split(',')
    test_datasets = args.test.strip('[]').split(',')

    # Train val test splits (only used on case of single dataset)
    train_val_test_split = ast.literal_eval(args.train_val_test_split)
    
    shuffle = bool(args.shuffle) # To boolean

    # Initialize folder paths
    path_models = f'Trained_models/{"finetune" if args.fine_tune_classification else "pre_trained"}/'
    path_best_models = f'best_models/{"finetune" if args.fine_tune_classification else "pre_trained"}/'
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()  else "cpu")

    # Finding the datasets in the training/validation sets.
    data_include = ''
    if 'DTU' in test_datasets:
        data_include += '_DTU'
    if 'WESAD' in test_datasets:
        data_include += '_WESAD'
    if 'ROAD' in test_datasets:
        data_include += '_ROAD'
    if 'ADARP' in test_datasets:
        data_include += '_ADARP'
    if 'WA' in test_datasets:
        data_include += '_WA'
    if set(['ADARP','DTU','ROAD','WESAD']) == set(train_val_datasets):
        data_include = '_All_Data'


    print('Initializing Data\n')

    duration = 60*args.minutes #duration of signal, 1min, 4min, 5min and 6min implemented
    if args.data_load == 'multi': #to include several datasets
        (X_train, X_val, X_test, 
         Y_train, Y_val, Y_test, 
         Y_user_train, Y_user_val, Y_user_test, 
         X_train_SIM, X_val_SIM) = get_data_load( 
            test = test_datasets, data_load=args.data_load, train_val = train_val_datasets, 
            split = args.train_val_split, duration = duration, BW_filter = args.BW_filter, 
            seed = args.seed, include_SIM = bool(args.include_SIM), before = args.before, SIM = args.additional_model_details)
    elif args.data_load == 'personal':
        (X_train, X_val, X_test, 
        Y_train, Y_val, Y_test, 
        Y_user_train, Y_user_val, Y_user_test, 
        index_train, index_val, index_test,
        X_train_personal, X_val_personal, X_test_personal, 
        Y_train_personal, Y_val_personal, Y_test_personal, 
        Y_user_train_personal, Y_user_val_personal, Y_user_test_personal, 
        index_train_personal, index_val_personal, index_test_personal) = get_data_load( 
                test = test_datasets, data_load=args.data_load, train_val = None, split = args.train_val_split,#skift her, 
                duration = duration, BW_filter = args.BW_filter, seed = args.seed, before = args.before, test_id = args.ID)
    elif args.data_load == 'single': # to include one only
        (X_train, X_val, X_test, 
         Y_train, Y_val, Y_test, 
         Y_user_train, Y_user_val, Y_user_test, 
         index_train, index_val, index_test) = get_data_load( 
                test = test_datasets, data_load=args.data_load, train_val = None, split = train_val_test_split, 
                duration = duration, BW_filter = args.BW_filter, seed = args.seed, before = args.before)
    elif args.data_load == 'single_SIM': # to include one only
        (X_train, X_val, X_test, 
         Y_train, Y_val, Y_test, 
         Y_user_train, Y_user_val, Y_user_test, 
         index_train, index_val, index_test, 
         X_train_SIM, X_val_SIM) = get_data_load( 
                test = test_datasets, data_load=args.data_load, train_val = None, split = train_val_test_split, 
                duration = duration, BW_filter = args.BW_filter, seed = args.seed, before = args.before, SIM = args.additional_model_details)
    else:
        raise Exception(f'Data load {args.data_load} not implemented.')
    Y_train_activity, Y_val_activity, Y_test_activity = None, None, None
    # Filter data classified as physical activity (if specified).
    if args.activity_classification:
        assert not args.data_load == 'multi'
        if test_datasets != ['DTU']:
            # Get indexes for physical activity
            (X_train, X_val, X_test, 
                Y_train, Y_val, Y_test, 
                Y_user_train, Y_user_val, Y_user_test) = activity_filter(X_train, X_val, X_test, 
                                                                        Y_train, Y_val, Y_test, 
                                                                        Y_user_train, Y_user_val, Y_user_test, 
                                                                        *get_activity_class(index_train = index_train, index_val = index_val, index_test = index_test, data_set=test_datasets[0], STD = True, 
                                                                    duration=duration, before = args.before, threshold = [40], SAMPLE_RATE=32, seed = args.seed))
    else:
        if test_datasets != ['DTU'] and args.data_load != 'multi':
            # Get indexes for physical activity
            Y_train_activity, Y_val_activity, Y_test_activity = get_activity_class(index_train = index_train, index_val = index_val, index_test = index_test, data_set=test_datasets[0], STD = True, 
                        duration=duration, before = args.before, threshold = [40], SAMPLE_RATE=32, seed = args.seed)

    if args.standardize: # Standardize if specified
        X_train, X_val, X_test = get_standardize(X_train, X_val, X_test, 
                                                method=args.stan_method, dim=args.stan_dim, 
                                                y_subject_train=Y_user_train, y_subject_val=Y_user_val, 
                                                y_subject_test=Y_user_test)
        
    # Convert to torch
    Y_train = torch.tensor(Y_train).int()
    Y_val = torch.tensor(Y_val).int()
    Y_test = torch.tensor(Y_test).int()
    Y_user_train = torch.tensor(Y_user_train)
    Y_user_test = torch.tensor(Y_user_test)
    Y_user_val = torch.tensor(Y_user_val)

    # Gives 4Hz and 64Hz data as torch.tensors
    X_train_freq, X_val_freq, X_test_freq = get_freq(X_train, X_val, X_test, duration)

    # Converts training data to tensor (for reconstruction)
    X_train, X_val, X_test = convert_X_tensor(X_train, X_val, X_test) 

    # Standardized X data (with specified method)

    #Simulation data is unlabeled thus separate X data must be created for autoencoder
    if args.include_SIM:
        X_train_SIM, X_val_SIM, _ = get_standardize(X_train_SIM, X_val_SIM, X_test, 
                                            method=args.stan_method, dim=args.stan_dim, 
                                            y_subject_train=Y_user_train, y_subject_val=Y_user_val, 
                                            y_subject_test=Y_user_test)
        X_train_freq_SIM = get_freq_single(X_train_SIM, duration)
        X_val_freq_SIM = get_freq_single(X_val_SIM, duration)
        X_train_SIM = convert_X_tensor_single(X_train_SIM) 
        X_val_SIM = convert_X_tensor_single(X_val_SIM) 
        X_data = [X_train_freq_SIM, X_val_freq_SIM, X_test_freq]
        original_data = [X_train_SIM, X_val_SIM, X_test]
    else: #If no simulation data. List are created of original data for training loop function.
        X_data = [X_train_freq, X_val_freq, X_test_freq]
        original_data = [X_train, X_val, X_test]

    print('Data initialized\n')

    # Number of signal per frequency. Relevant if you include subset of signal for a given frequency.
    NumFeats = [x.shape[-1] for x in X_train_freq]

    # Sequence length for each signal (sample rate * duration).
    seq_length = [x.shape[1] for x in X_train_freq]

    # Architecture of model
    architecture = args.architecture

    # Configuration for model
    config = model_config(
            NumComps = len(nodes_per_freq),
            NumFeats = NumFeats,
            nodes_before_last_layer = 500,
            nodes_per_freq = nodes_per_freq,
            dropout = args.dropout,
            architecture = architecture,
            NumLayers = args.num_layers,
            CNNKernelSize = kernel_size,
            Out_Nodes = np.sum([i.shape[1] for i in X_train]),
            seq_length = seq_length,
            R_U = args.R_U)

    #### For txt report ####
    #region txt file report initialization
    #bn = 'BN_' + str(nodes_per_freq[0]) + '_' + str(nodes_per_freq[1])
    bn = f'BN_{"_".join(str(i) for i in nodes_per_freq)}'
    if args.fine_tune_classification:
        run_folder = 'runs_finetune/'
    elif test_datasets == ['WA']:
        run_folder = 'runs_WA/'
    else:
        run_folder = 'runs_auto/'
    run_folder += f'{architecture}_{bn}_{data_include}_{duration}sec{"_bf"+str(args.before) if args.before else ""}{"_AC" if args.activity_classification else ""}{"_personal" if args.data_load == "personal" else ""}{"_stratify" if args.stratify else ""}{"_nontrained" if not args.train_classification else ""}{args.folder_arg}/'
    # Name of autoencoder (for saving)
    model_name = f'autoencoder_{architecture}_{bn}_{duration}sec{data_include}{"_AC" if args.activity_classification else ""}{"_ID"+str(args.ID) if args.ID else ""}{"_stratify" if args.stratify else ""}{args.additional_model_details}{"_finetuned" if args.fine_tune_classification else ""}'
    time_run = str(datetime.now().strftime('%m-%d_%H_%M'))
    folder_runs = f'{run_folder}run_{args.run}/{str(args.ID)+"/" if args.ID else ""}'
    Path(folder_runs).mkdir(parents=True, exist_ok=True)
    Path(path_best_models).mkdir(parents=True, exist_ok=True)
    Path(path_models).mkdir(parents=True, exist_ok=True)
    txt_output = ''
    txt_output += f'Model: {architecture}\n'
    txt_output += f'BN: {bn}\n'
    txt_output += f'nodes_per_freq: {nodes_per_freq}\n'
    if args.txt_detail:
        txt_output += f'{args.txt_detail}\n'
    txt_output += f'duration: {duration} sec\n'
    txt_output += f'Data: {data_include}\n'
    txt_output += f'Number of layers: {config.NumLayers}\n'
    txt_output += f'Reverse U-Net: {config.R_U}\n'
    txt_output += f'Use extra linear layer: {args.UseExtraLinear}\n'
    if args.UseExtraLinear:
        txt_output += f'Number nodes before last layer: {config.nodes_before_last_layer}\n'
    txt_output += f'Kernel size: {config.CNNKernelSize}\n'
    txt_output += f'Dropout: {config.dropout}\n'
    txt_output += f'delta (relative): {args.delta}\n'
    txt_output += f'patience: {args.patience}\n'
    #endregion

    # Defining autoencoder
    model = getModel(config)
    model = model.to(device)

    # Load saved autoencoder to be retrained (if specified).
    if args.continue_train_autoencoder:
        model.load_state_dict(torch.load(path_best_models+model_name, map_location = device))

    
    # Defining training parameters for autoencoder
    autoencoder_config = auto_config(
        model = model,
        model_name = model_name,
        device = device,
        data = X_data,
        original_data = original_data,
        n_epochs = args.epochs,
        batch_size = args.batch_size,
        lr = args.lr,
        patience = args.patience,
        delta = args.delta,
        optimizer_name = 'Adam',
        lr_scheduler_name = 'ReduceLROnPlateau',
        criterion_name = 'MSELoss',
        seed = args.seed,
        shuffle = shuffle,
        path_models = path_models,
        folder_runs = folder_runs,
        txt_output = txt_output,
        run = args.run
    )

    # Training autoencoder
    if args.train_autoencoder:
        print('\nAutoencoder train\n')
        model, txt_output = train_autoencoder(autoencoder_config)
    # else:
    #     # In the case classification is fine-tuned. Autoencoder does not need to be trained or loaded.
    #     if not args.fine_tune_classification:
    #         model.load_state_dict(torch.load(path_models+model_name))

    # Defining classification model
    c_model = getModel(config, model)#, nodes_before_last_layer=int(sum(nodes_per_freq)/2)
    c_model = c_model.to(device) #sending to device

    # Name of classification model (for saving)
    classification_model = f'classification_{architecture}_{bn}_{duration}sec{data_include}{"_bf"+str(args.before) if args.before else ""}{"_AC" if args.activity_classification else ""}{"_ID"+str(args.ID) if args.ID else ""}{"_ID"+str(args.ID) if args.ID else ""}{"_stratify" if args.stratify else ""}{args.additional_model_details}{"_finetuned" if args.fine_tune_classification else ""}'


    if args.continue_train_classification: # Retrain classification (not fine-tuning).
        c_model.load_state_dict(torch.load(path_best_models+classification_model, map_location = device))
    elif args.fine_tune_classification:# or (args.train_classification):
        print('\nFine-tuning\n')
        # Model are not pre trained on prediction lead time != 0, thus no model exists with a name included "bf_x".
        # Additionally, they aren't called "_finetuned"
        data_include_temp = '_All_Data' if data_include == '_WA' else data_include
        classification_model_temp = f'classification_{architecture}_{bn}_{duration}sec{data_include_temp}{"_stratify" if args.stratify and args.train_autoencoder else ""}{args.additional_model_details}'
        # Loading best (folder best_models) classification model for pre-training. 
        c_model.load_state_dict(torch.load(f'best_models/pre_trained/{classification_model_temp}', map_location = device))
        del classification_model_temp, data_include_temp

    #self supervised 
    if args.train_sub_recog_self:
        train_sub_model()
        
    if args.freeze_CNN:
        c_model.freqmodels.requires_grad_(False)
    

    data_class = [E4Data_freq_stratify(X, device, Y, Y_user) if args.stratify else E4Data_freq(X, device, Y) 
                  for X, Y, Y_user in zip([X_train_freq,X_val_freq,X_test_freq], 
                                          [Y_train, Y_val, Y_test], 
                                          [Y_user_train, Y_user_val, Y_user_test])]

    # Defining training parameters
    classification_config = classi_config(
        model = c_model,
        model_name = classification_model,
        device = device,
        data = [X_train_freq, X_val_freq, X_test_freq],
        y_data = [Y_train, Y_val, Y_test],
        data_class = data_class,
        n_epochs = args.epochs,
        batch_size = args.batch_size,
        lr = args.lr/10,
        patience = args.patience,
        delta = args.delta,
        optimizer_name = 'Adam',
        lr_scheduler_name = 'LambdaLR',
        criterion_name = 'BCEWithLogitsLoss',
        seed = args.seed,
        shuffle = shuffle,
        path_models = path_models,
        folder_runs = folder_runs,
        txt_output = txt_output,
        run = args.run
    )

    # Training classification model
    if args.train_classification:
        print('\nClassification train\n')
        c_model, txt_output = train_classification(classification_config)
    else:
        # In case just predictions need to be made.
        # To use pre trained classification model fine-tune classification should be False. To use fine-tuned set it to True.
        data_include_temp = '_All_Data' if data_include == '_WA' else data_include
        classification_model_temp = f'classification_{architecture}_{bn}_{duration}sec{data_include_temp}{"_stratify" if args.stratify and args.train_autoencoder else ""}{args.additional_model_details}'
        # Loading best (folder best_models) classification model for pre-training. 
        c_model.load_state_dict(torch.load(f'best_models/pre_trained/{classification_model_temp}', map_location = device)) # temp notice only pretrained

    # Classification prediction
    #region y_pred, accuracy, and F1-score
    # Create folder for saving prediction (if specified)
    if args.save_pred:
        pred_folder = f'predictions/{classification_config.model_name}{"_nontrained" if not args.train_classification else ""}/run_{str(args.run)}/'
        pred_name = f'{pred_folder}{classification_config.model_name}{"_nontrained" if not args.train_classification else ""}'
        Path(pred_folder).mkdir(parents=True, exist_ok=True)

    y_train, y_pred_train = predict(c_model, X_train_freq, Y_train, device, activity_index=Y_train_activity, save_pred = f'{pred_name+"_train_"+str(time_run) if args.save_pred else ""}')
    y_val, y_pred_val = predict(c_model, X_val_freq, Y_val, device, activity_index=Y_val_activity, save_pred = f'{pred_name+"_val_"+str(time_run) if args.save_pred else ""}')
    y_test, y_pred_test = predict(c_model, X_test_freq, Y_test, device, activity_index=Y_test_activity, save_pred = f'{pred_name+"_test_"+str(time_run) if args.save_pred else ""}')

    # Accuracy
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_val = accuracy_score(y_val, y_pred_val)
    acc_test = accuracy_score(y_test, y_pred_test)

    # F1 score
    f1_train = f1_score(y_train, y_pred_train)
    f1_val = f1_score(y_val, y_pred_val)
    f1_test = f1_score(y_test, y_pred_test)

    print('Accuracy score:')
    print(f"Train: {acc_train:.2}")
    print(f"Val: {acc_val:.2}")
    print(f"Test: {acc_test:.2}")
    print('F1-score:')
    print(f'Train: {f1_train:.2}')
    print(f'Val: {f1_val:.2}')
    print(f'Test: {f1_test:.2}')
    #endregion

    # For txt report 
    txt_output += (f'Accuracy: \nTrain: {acc_train:.2}\nVal: {acc_val:.2}\nTest: {acc_test:.2}\n')
    txt_output += f'F1-Score: \nTrain: {f1_train:.2} \nVal: {f1_val:.2} \nTest: {f1_test:.2}'

    # Fine-tunening classification model to an individual (if personal specified)
        

    
    if args.ID:
        if args.train_classification:
            print('Personalized training\n')
            Y_train_activity_personal, Y_val_activity_personal, Y_test_activity_personal = None, None, None
            if args.activity_classification:
                if test_datasets != ['DTU']:
                    (X_train_personal, X_val_personal, X_test_personal, 
                    Y_train_personal, Y_val_personal, Y_test_personal, 
                    Y_user_train_personal, Y_user_val_personal, Y_user_test_personal) = activity_filter(X_train_personal, X_val_personal, X_test_personal, 
                                                                                            Y_train_personal, Y_val_personal, Y_test_personal, 
                                                                                            Y_user_train_personal, Y_user_val_personal, Y_user_test_personal, 
                                                                                            *get_activity_class(index_train = index_train_personal, index_val = index_val_personal, index_test = index_test_personal, data_set=test_datasets[0], STD = True, 
                                                                                        duration=duration, before = args.before, threshold = [40], SAMPLE_RATE=32, seed = args.seed))
            else:
                if test_datasets != ['DTU']:
                    Y_train_activity_personal, Y_val_activity_personal, Y_test_activity_personal = get_activity_class(index_train = index_train_personal, index_val = index_val_personal, index_test = index_test_personal, data_set=test_datasets[0], STD = True, 
                                                                                        duration=duration, before = args.before, threshold = [40], SAMPLE_RATE=32, seed = args.seed)
            # Standardized X data (with specified method)
            if args.standardize: # Standardize if specified
                X_train_personal, X_val_personal, X_test_personal = get_standardize(X_train_personal, X_val_personal, X_test_personal, 
                                                        method=args.stan_method, dim=args.stan_dim, 
                                                        y_subject_train=Y_user_train_personal, y_subject_val=Y_user_val_personal, 
                                                        y_subject_test=Y_user_test_personal)
            # Convert to torch
            Y_train_personal = torch.tensor(Y_train_personal)
            Y_val_personal = torch.tensor(Y_val_personal)
            Y_test_personal = torch.tensor(Y_test_personal)
            Y_user_train_personal = torch.tensor(Y_user_train_personal)
            Y_user_test_personal = torch.tensor(Y_user_test_personal)
            Y_user_val_personal = torch.tensor(Y_user_val_personal)

            # Gives 4Hz and 64Hz data as torch.tensors
            X_train_freq_personal, X_val_freq_personal, X_test_freq_personal = get_freq(X_train_personal, X_val_personal, X_test_personal, duration)
            classification_config.model = c_model
            classification_config.data = [X_train_freq_personal, X_val_freq_personal, X_test_freq_personal]
            classification_config.y_data = [Y_train_personal, Y_val_personal, Y_test_personal]
            classification_config.txt_output = txt_output
            classification_config.model_name = classification_model
            c_model, txt_output = train_classification(classification_config)
        if args.save_pred:
            pred_folder = f'predictions/{classification_config.model_name}{"_nontrained" if not args.train_classification else ""}/run_{str(args.run)}/personalized/'
            pred_name = f'{pred_folder}{classification_config.model_name}{"_nontrained" if not args.train_classification else ""}'
            Path(pred_folder).mkdir(parents=True, exist_ok=True)

        y_train, y_pred_train = predict(c_model, X_train_freq_personal, Y_train_personal, device, activity_index=Y_train_activity_personal, save_pred = f'{pred_name+"_train_"+str(time_run) if args.save_pred else ""}')
        y_val, y_pred_val = predict(c_model, X_val_freq_personal, Y_val_personal, device, activity_index=Y_val_activity_personal, save_pred = f'{pred_name+"_val_"+str(time_run) if args.save_pred else ""}')
        y_test, y_pred_test = predict(c_model, X_test_freq_personal, Y_test_personal, device, activity_index=Y_test_activity_personal, save_pred = f'{pred_name+"_test_"+str(time_run) if args.save_pred else ""}')

        # Accuracy
        acc_train2 = accuracy_score(y_train, y_pred_train)
        acc_val2 = accuracy_score(y_val, y_pred_val)
        acc_test2 = accuracy_score(y_test, y_pred_test)

        # F1 score
        f1_train2 = f1_score(y_train, y_pred_train)
        f1_val2 = f1_score(y_val, y_pred_val)
        f1_test2 = f1_score(y_test, y_pred_test)

        print('Accuracy score:')
        print(f"Train: {acc_train2:.2}")
        print(f"Val: {acc_val2:.2}")
        print(f"Test: {acc_test2:.2}")
        print('F1-score:')
        print(f'Train: {f1_train2:.2}')
        print(f'Val: {f1_val2:.2}')
        print(f'Test: {f1_test2:.2}')

        txt_output += (f'Accuracy: \nTrain: {acc_train2:.2}\nVal: {acc_val2:.2}\nTest: {acc_test2:.2}\n')
        txt_output += f'F1-Score: \nTrain: {f1_train2:.2} \nVal: {f1_val2:.2} \nTest: {f1_test2:.2}'
    
    txt_name = f'{classification_model}{"_nontrained" if not args.train_classification else ""}_{time_run}'
    #txt_name = f'{architecture}_{bn}_{config.NumLayers}{R_U}_{duration}sec{data_include}{"_bf"+str(args.before) if args.before else ""}{"_AC" if args.activity_classification else ""}{"_ID"+args.ID if args.ID else ""}_{time_run}'

    # Saving txt report for each run
    with open(folder_runs+txt_name, 'w') as f:
        f.write(txt_output)

    #region txt data frame with accuracy and f1 score for every run by model architecture
    tot_str = f'Run {args.run}, {acc_train:.2}, {f1_train:.2}, {acc_val:.2}, {f1_val:.2}, {acc_test:.2}, {f1_test:.2}\n' 
    if args.ID:
        tot_str += f'Run {args.run} ID {args.ID}, {acc_train2:.2}, {f1_train2:.2}, {acc_val2:.2}, {f1_val2:.2}, {acc_test2:.2}, {f1_test2:.2}\n' 
    with open(run_folder+'total.txt', 'a') as file:
        # Append new content to the file
        if file.tell()==0:
            file.write(', Acc_Train, F1_Train, Acc_Val, F1_Val, Acc_Test, F1_Test\n')
        file.write(tot_str) # Saving accuracy and f1 scores in one txt file for each model architecture
    #endregion

    #region testing of the model run should be saved or not
    # Load data from previous runs
    run_data = pd.read_csv(run_folder+'total.txt', sep=", ", index_col=0, engine='python')

    # Check if current model is best model
    # Autoencoder
    if args.continue_train_autoencoder or args.train_autoencoder:
        if args.save_first_run and args.run==1: # Saves model for run 1 (if specified)
            print('First run saved Autoencoder\n')
            torch.save(model.state_dict(),path_best_models+model_name)
        elif len(run_data['F1_Val']) > 1: # If more than 
            if not args.save_first_run:
                if ((run_data['F1_Val'][-1]+run_data['Acc_Val'][-1]) > max(run_data['F1_Val'][:-1]+run_data['Acc_Val'][:-1])):
                    print('New best model saved Autoencoder\n')
                    torch.save(model.state_dict(),path_best_models+model_name)
            elif ((run_data['F1_Val'][-1]+run_data['Acc_Val'][-1]) > max(run_data['F1_Val'][-args.run:-1]+run_data['Acc_Val'][-args.run:-1])):
                print('New best model saved Autoencoder\n')
                torch.save(model.state_dict(),path_best_models+model_name)
        else: # If no previous run data 
            print('First run best model saved Autoencoder\n')
            torch.save(model.state_dict(),path_best_models+model_name)

    # Classification
    if args.train_classification or args.continue_train_classification or args.fine_tune_classification:
        if args.save_first_run and args.run==1: # Saves model for run 1 (if specified)
            print('First run saved Classification\n')
            torch.save(c_model.state_dict(),path_best_models+classification_model)
        elif len(run_data['F1_Val']) > 1: # If more than 
            if not args.save_first_run:
                if ((run_data['F1_Val'][-1]+run_data['Acc_Val'][-1]) > max(run_data['F1_Val'][:-1]+run_data['Acc_Val'][:-1])):
                    print('New best model saved Classification\n')
                    torch.save(c_model.state_dict(),path_best_models+classification_model)
            elif ((run_data['F1_Val'][-1]+run_data['Acc_Val'][-1]) > max(run_data['F1_Val'][-args.run:-1]+run_data['Acc_Val'][-args.run:-1])):
                print('New best model saved Classification\n')
                torch.save(c_model.state_dict(),path_best_models+classification_model)
        else: # If no previous run data 
            print('First run best model saved Classification\n')
            torch.save(c_model.state_dict(),path_best_models+classification_model)
    #endregion

    # Moving this training down, as it is not used in main training
    # Selfsupervised learning for subject recognition 
    def train_sub_model():
        vec_dim = torch.max(torch.tensor([torch.max(Y_user_train),torch.max(Y_user_val),torch.max(Y_user_test)]))
        sub_model_name = f'per_sim_{architecture}_{bn}_{duration}sec{data_include}{"_bf"+str(args.before) if args.before else ""}{args.additional_model_details}'
        sub_model = getModel(config, c_model, NumClasses=vec_dim, classification=False)
        sub_model = sub_model.to(device)
        sub_config = sub_diff_config(
            model = model,
            model_name = sub_model_name,
            device = device,
            data = [X_train_freq, X_val_freq, X_test_freq],
            y_data = [Y_user_train, Y_user_val, Y_user_test],
            n_epochs = args.epochs,
            batch_size = int(len(Y_user_train)/2),
            lr = args.lr,
            patience = args.patience,
            delta = args.delta,
            optimizer_name = 'Adam',
            lr_scheduler_name = 'ReduceLROnPlateau',
            criterion_name = 'BYOLoss',
            shuffle = False,
            path_models = path_models,
            folder_runs = folder_runs,
            run = args.run
        )
        sub_model = train_subject(sub_config)
        c_model = getModel(config, sub_model, nodes_before_last_layer=int(sum(nodes_per_freq)/2))
        c_model = c_model.to(device)

def parse_arguments():
    '''
    Parse arguments via command-line.

    This script does stress prediction.

    Example usage:
        python main.py '"[DTU]"' --train_val '"[ADARP,ROAD,WESAD]"'
    '''
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument("test", help="Dataset in val and test", type=str)
    parser.add_argument("--train_val", help="Datasets in train", type=str, default = '')
    parser.add_argument("--data_load", help="What type of data to be loaded. Several, single, or specific person (requires ID). Default: single", type=str, choices = ['multi', 'single', 'personal','single_SIM'], default = 'single')
    parser.add_argument("--train_val_split", help="Train, and Val splits. For single dataset", type=int, default = 0.2)
    parser.add_argument("--train_val_test_split", help="Train, Val, and Test splits. For single dataset", type=str, default = str([0.8,0.1,0.1]))
    parser.add_argument("--include_SIM", help="If simulated data should be included. Default: 0 (No)", type=int, choices=[0,1], default=0)
    parser.add_argument("--minutes", help="Interval length in minutes", type=int, default=5)
    parser.add_argument("--before", help="Amount of seconds before a stress episode we predict.", type=int, default=0)
    parser.add_argument("--standardize", help="If data should be standardized. Default: 1 (Yes)", type=int, choices=[0,1], default=1)
    parser.add_argument("--stan_method", help="Standardization method. Default: 'obs'", type=str, choices=['obs','sub'], default='obs')
    parser.add_argument("--stan_dim", help="Dimensions to standardize. Specify dimensions to standardize with list of dim like [1,3]. Default: 'all'", type=str, default='all')
    parser.add_argument("--BW_filter", help="Uses BW filter on data. Default: 1 (Yes)", type=int, choices=[0,1], default=1)
    parser.add_argument('-AC',"--activity_classification", help="If data should be filtered by physical activity. Default: 0 (No)", type=int, choices=[0,1], default=0)
    parser.add_argument("--ID", help="ID for person to be fine-tuned on. Default: 0 (None)", type=int, default=0)
    parser.add_argument("-pf", "--personalized_finetune", help="If model should be retrained to specific person by ID. Requires retrain=True. Default: 0 (No)", type=int, choices= [0,1], default=0)
    parser.add_argument("--stratify", help="If data should be stratified by user. Default: 0 (No)", type=int, choices= [0,1], default=0)
    

    # Model architecture
    parser.add_argument("--architecture", help="Model architecture. Default: 'CNN'", type=str, default='CNN')
    parser.add_argument("--n1_n2", help="Nodes in bottleneck for 4Hz and 64Hz. Default 50 4HZ, 25 64Hz.", type=str, default=str([50,25]))
    parser.add_argument("-nl", "--num_layers", help="Number of hidden layers. Default: 3", type=int, default=3)
    parser.add_argument("-UEL", "--UseExtraLinear", help="Use extra layer in last linear layer. Default: 1 (Yes)", type=int, choices=[0,1], default=1)
    parser.add_argument("--R_U", help="If model should have a reverse U-Net architecture. Default: 0 (No).", type=int, choices=[0,1], default=0)
    parser.add_argument("-ks", "--kernel_size", help="Sizes of kernels for each layer. List with kernelsize for each layer. Default: 3 (for each layer).", type=str, default=str([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]))
    parser.add_argument("--dropout", help="Dropout rate. Default 0.2.", type=float, default=0.2)
    parser.add_argument("--freeze_CNN", help="If CNN layer (encoder) should be freezed for classification training. Default: 0 (No)", type=int, choices=[0,1], default=0)

    # Training arguments
    parser.add_argument("--patience", help="Number of epochs with no improvement before termination. Default: 15.", type=int, default=15)
    parser.add_argument("--lr", help="Learning rate: Default: 0.1", type=float, default=0.1)
    parser.add_argument("--epochs", help="Number of epochs. Default: 200.", type=int, default=200)
    parser.add_argument("--batch_size", help="Batch Size. Default: 64.", type=int, default=64)
    parser.add_argument("--delta", help="Relative difference in improvement needed. Default: 0.01 (1%).", type=int, default=0.01)
    parser.add_argument("--train_autoencoder", help="Train autoencoder. Default: 1 (Yes)", type=int, choices=[0,1], default=1)
    parser.add_argument("--train_classification", help="Train classification.  Default: 1 (Yes)", type=int, choices=[0,1], default=1)
    parser.add_argument("-tsrs", "--train_sub_recog_self", help="Train classification.  Default: 0 (No)", type=int, choices=[0,1], default=0)
    parser.add_argument("--continue_train_autoencoder", help="Retrain autoencoder. Default: 0 (No)", type=int, choices=[0,1], default=0)
    parser.add_argument("--continue_train_classification", help="Continue train classification.  Default: 0 (No)", type=int, choices=[0,1], default=0)
    parser.add_argument("-ftc", "--fine_tune_classification", help="If fine-tuning should be done instead (trains only on test data). Requires pre-trained models in (best_models/pre_trained).  Default: 0 (No).", type=int, choices=[0,1], default=0)
    parser.add_argument("--shuffle", help="If training data should be randomly shuffled. Mainly used for fine-tunening.  Default: 0 (No).", type=int, choices=[0,1], default=0)

    # Naming of folder arguments
    parser.add_argument("--folder_arg", help="Extra arguments for folder name.", type=str, default='')
    parser.add_argument("-sfr", "--save_first_run", help="Saves model in first run. Will overwrite previous best model.", type=int, default=0)
    parser.add_argument("-amd", "--additional_model_details", help="Add additional model detail to model name.", type=str, default='')
    parser.add_argument("-amp","--activity_model_path", help="Path and name of activity model", type=str, default='')
    parser.add_argument("--txt_detail", help="Additional details to add to txt report.", type=str, default='')

    # Meta arguments
    parser.add_argument("--run", help="An integer for the run", type=int, default=1)
    parser.add_argument("--seed", help="Seed for file. Default: 123", type=int, default=123)
    parser.add_argument("--save_pred", help="If predictions with true values should be saves. Default: 1 (Yes)", type=int, default=0)

    # parse all arguments to python
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())