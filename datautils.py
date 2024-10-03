import logging
import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle

import torch

from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_UCR(dataset):
    train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels

    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_UEA(dataset):
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]

    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)

    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y


def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')
    if univar:
        data = data[: -1:]

    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)

    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


def load_forecast_csv(name, univar=False):
    """
    Loads and processes time series data for forecasting tasks, supporting both the pre-processed
    Online Retail II dataset and existing datasets like ETTh1, ETTm1, and electricity.

    Parameters:
    name (str): The name of the dataset file (without the .csv extension).
    univar (bool): Whether to load the univariate version of the data.

    Returns:
    tuple: data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols
    """
    # Try loading with 'date' as index, if it fails, try with 'InvoiceDate' to load online retail data
    try:
        data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    except ValueError:
        data = pd.read_csv(f'datasets/{name}.csv', index_col='InvoiceDate', parse_dates=True)

    # Ensure index is parsed as datetime
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        data.index = pd.to_datetime(data.index)

    # Extract time features for the date/index column
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]

    # Handle univariate or multivariate cases
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]

    # Convert data to numpy array
    data = data.to_numpy()

    # Define train, validation, and test splits based on dataset
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        # Default case for other or custom datasets
        train_slice = slice(0, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), len(data))

    # Normalise data
    scaler = None
    if name == 'ts2vec_online_retail_II_data' or name == 'restructured_ts2vec_online_retail':
        scaler = MinMaxScaler().fit(data[train_slice])
    else:
        scaler = StandardScaler().fit(data[train_slice])

    data = scaler.transform(data)

    # Reshape data based on dataset structure
    if name in 'electricity':
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0) # Single instance case

    if n_covariate_cols > 0:
        if name == 'ts2vec_online_retail_II_data' or name == 'restructured_ts2vec_online_retail':
            dt_scaler = MinMaxScaler().fit(dt_embed[train_slice])
        else:
            dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        # Concatenating the time embeddings to the data
        ''' NOTE: 
        The np.repeat(dt_embed, data.shape[0], axis=0) function is used to repeat the time embeddings 
        for each instance only in the case of the 'electricity' dataset. This ensures that the time 
        embeddings are correctly aligned with the data instances'''
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)

    if name in ('ETTh1', 'ETTh2', 'electricity'):
        pred_lens = [24, 48, 168, 336, 720]
    elif name == 'ts2vec_online_retail_II_data':
        pred_lens = [1, 2, 3, 4, 5]
    else:
        pred_lens = [24, 48, 96, 288, 672]

    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_anomaly(name):
    res = pkl_load(f'datasets/{name}.pkl')
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
        res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
        res['delay']


def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data

#-----------------------------------------------------------------------------
def _get_time_features(dt):
    return np.stack([
        dt.dayofweek.to_numpy(),  # Day of the week
        dt.day.to_numpy(),        # Day of the month
        dt.dayofyear.to_numpy(),  # Day of the year
        dt.month.to_numpy(),      # Month
        dt.to_series().apply(lambda x: x.strftime("%V")).astype(int).to_numpy(), # Week of the year
    ], axis=1).astype(np.float)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def load_online_retail(name):
#     """
#     Loads and preprocesses the Online Retail dataset for forecasting tasks.
#     Ensures both Price, Quantity, and customer embeddings are included throughout. """
#
#     Returns:
#         train_data: Dictionary mapping customer_id to their training data (DataFrame).
#         valid_data: Dictionary mapping customer_id to their validation data (DataFrame).
#         test_data: Dictionary mapping customer_id to their test data (DataFrame).
#         customer_embeddings: Fixed embeddings for each customer ID.
#         customer_id_to_index: Mapping from customer IDs to embedding indices.
#         scaler: Fitted scaler for data normalization.
#     """

     # Load data
    data = pd.read_csv(f'datasets/{name}.csv', index_col='InvoiceDate', parse_dates=True)

    # Convert 'InvoiceDate' to datetime if not already done
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        data.index = pd.to_datetime(data.index)

    # Remove rows with any None or NaN values
    data.dropna(inplace=True)

    # Remove rows with negative 'Quantity' or 'Price' values
    data = data[(data['Quantity'] > 0) & (data['Price'] > 0)]
    logging.info(f"Data shape after removing negative values: {data.shape}")

    # Remove Invoice column
    data.drop(columns=['Invoice'], inplace=True)

    # Rename 'Customer ID' to 'CustomerID'
    if 'Customer ID' in data.columns:
        data.rename(columns={'Customer ID': 'CustomerID'}, inplace=True)

    if 'CustomerID' not in data.columns:
        raise KeyError("The 'CustomerID' column is missing from the dataset.")
    logging.info(f"Data columns and shape: {data.columns, data.shape}")

    # Extract customerIDs and create numerical mapping
    customer_ids = data['CustomerID'].unique()
    logging.info(f"Unique customer ID count: {customer_ids.shape[0]}")
    customer_id_to_index = {cid: idx for idx, cid in enumerate(customer_ids)}
    logging.info(f"Unique customer ID to index count: {len(customer_id_to_index)}")

    # Create fixed embeddings for each unique customer ID
    customer_embeddings = torch.nn.Embedding(len(customer_ids), 4)
    customer_embeddings.weight.requires_grad = False  # Fixed embeddings

    # Map customer IDs to their embeddings
    customer_embeddings_tensor = customer_embeddings(torch.tensor(data['CustomerID'].map(customer_id_to_index).values))
    logging.info(f"Customer embeddings shape: {customer_embeddings_tensor.shape}")
    logging.info(f"Customer embeddings: {customer_embeddings_tensor}")

    # Replace 'CustomerID' column with the corresponding customer embedding
    customer_embeddings_expanded = customer_embeddings_tensor.detach().numpy()
    logging.info(f"Customer embeddings expanded shape: {customer_embeddings_expanded.shape}")

    # Drop the original 'CustomerID' column
    data = data.drop(columns=['CustomerID'])

    # Extract time features for the date/index column
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1] + customer_embeddings_expanded.shape[-1]
    logging.info(f"Number of covariate columns: {n_covariate_cols}")
    logging.info(f"Time features shape: {dt_embed.shape}")

    # Convert the DataFrame to a NumPy array
    data = data.to_numpy()

    train_slice = slice(0, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), len(data))

    # Normalise data
    scaler = MinMaxScaler().fit(data[train_slice])
    data = scaler.transform(data)

    # Reshape data based on dataset structure
    data = np.expand_dims(data, 0) # Single instance case

    if n_covariate_cols > 0:
        dt_scaler = MinMaxScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        # Concatenating the time embeddings to the data
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)

        cid_scaler = MinMaxScaler().fit(customer_embeddings_expanded[train_slice])
        cid_embed = np.expand_dims(cid_scaler.transform(customer_embeddings_expanded), 0)
        # Concatenating the customer ID embeddings to the data
        data = np.concatenate([np.repeat(cid_embed, data.shape[0], axis=0), data], axis=-1)

    pred_lens = [1, 2, 3, 4, 5]

    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols



    # # Filter customerIDs with at least 5 records
    # customer_counts = data['CustomerID'].value_counts()
    # valid_customers = customer_counts[customer_counts >= 5].index
    # data = data[data['CustomerID'].isin(valid_customers)]
    #
    # # Group by CustomerID and sort by InvoiceDate
    # grouped = data.groupby('CustomerID').apply(lambda x: x.sort_values('InvoiceDate')).reset_index(drop=True)
    #
    # # Create time series data for each CustomerID
    # customer_data = {}
    # for customer_id, group in grouped.groupby('CustomerID'):
    #     group.set_index('InvoiceDate', inplace=True)
    #     customer_data[customer_id] = group[['CustomerID', 'Quantity', 'Price']]

    # Set other forecasting parameters
    # scaler = MinMaxScaler() # alternative to StandardScaler which avoid negative values
    # pred_lens = [1, 2, 3]
    # n_covariate_cols = 2
    #
    # return customer_data
