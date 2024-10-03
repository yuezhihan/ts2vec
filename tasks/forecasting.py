import numpy as np
import time
from . import _eval_protocols as eval_protocols

def generate_pred_samples(features, data, pred_len, drop=0):
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
        labels.reshape(-1, labels.shape[2]*labels.shape[3])

def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean()
    }

def eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, dataset_name = None):
    padding = 200

    t = time.time()
    all_repr = model.encode(
        data,
        causal=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    ts2vec_infer_time = time.time() - t

    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]

    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]

    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)

        t = time.time()
        lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t

        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        if dataset_name == 'ts2vec_online_retail_II_data':
            test_pred = test_pred.reshape(-1, 2)
            test_labels = test_labels.reshape(-1, 2)
        elif dataset_name == 'restructured_ts2vec_online_retail':
            test_pred = test_pred.reshape(-1, 4)
            test_labels = test_labels.reshape(-1, 4)
        else:
            ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
            test_pred = test_pred.reshape(ori_shape)
            test_labels = test_labels.reshape(ori_shape)

        if test_data.shape[0] > 1:
            test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
            test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
        elif dataset_name in ['ts2vec_online_retail_II_data','restructured_ts2vec_online_retail']:
            if dataset_name == 'ts2vec_online_retail_II_data':
                test_pred_inv = scaler.inverse_transform(test_pred.reshape(-1, 2)).reshape(test_pred.shape)
                test_labels_inv = scaler.inverse_transform(test_labels.reshape(-1, 2)).reshape(test_labels.shape)
            else: # restructured_ts2vec_online_retail
                test_pred_inv = scaler.inverse_transform(test_pred.reshape(-1, 4)).reshape(test_pred.shape)
                test_labels_inv = scaler.inverse_transform(test_labels.reshape(-1, 4)).reshape(test_labels.shape)

            # Remove NaN values from all datasets
            valid_indices = ~np.isnan(test_pred).any(axis=1) & ~np.isnan(test_pred_inv).any(axis=1) & ~np.isnan(test_labels).any(axis=1) & ~np.isnan(test_labels_inv).any(axis=1)
            test_pred = test_pred[valid_indices]
            test_pred_inv = test_pred_inv[valid_indices]
            test_labels = test_labels[valid_indices]
            test_labels_inv = test_labels_inv[valid_indices]

        else:
            test_pred_inv = scaler.inverse_transform(test_pred)
            test_labels_inv = scaler.inverse_transform(test_labels)

        out_log[pred_len] = {
            'norm': test_pred,
            'raw': test_pred_inv,
            'norm_gt': test_labels,
            'raw_gt': test_labels_inv
        }
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred, test_labels),
            'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }

        eval_res = {
            'ours': ours_result,
            'ts2vec_infer_time': ts2vec_infer_time,
            'lr_train_time': lr_train_time,
            'lr_infer_time': lr_infer_time
        }
    return out_log, eval_res

def eval_forecasting_customer_embed(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols):
    padding = 200

    t = time.time()
    all_repr = model.encode(
        data,
        causal=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    ts2vec_infer_time = time.time() - t

    # Extract representations (embeddings) for each time slice
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]

    # Extract data for each time slice
    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]

    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)

        t = time.time()
        lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t

        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        test_pred = test_pred.reshape(-1, 2)
        test_labels = test_labels.reshape(-1, 2)

        test_pred_inv = scaler.inverse_transform(test_pred.reshape(-1, 2)).reshape(test_pred.shape)
        test_labels_inv = scaler.inverse_transform(test_labels.reshape(-1, 2)).reshape(test_labels.shape)

        # NOTE: You can now use train_repr, valid_repr, and test_repr for unsupervised tasks (e.g., clustering, visualization)

        out_log[pred_len] = {
            'norm': test_pred,
            'raw': test_pred_inv,
            'norm_gt': test_labels,
            'raw_gt': test_labels_inv
        }
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred, test_labels),
            'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }

        eval_res = {
            'ours': ours_result,
            'ts2vec_infer_time': ts2vec_infer_time,
            'lr_train_time': lr_train_time,
            'lr_infer_time': lr_infer_time
        }

    return out_log, eval_res

