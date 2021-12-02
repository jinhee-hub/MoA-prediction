#Zhenhao Lu wrote this file

import numpy as np
import random
import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import copy
# import seaborn as sns

# from sklearn import preprocessing
from sklearn.metrics import log_loss
# from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import Datasets
import warnings

warnings.filterwarnings('ignore')

# Set HyperParameters
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NFOLDS = 5
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False

from sklearn.model_selection import KFold


class MultilabelStratifiedKFold:
    def __init__(self, n_splits=1):
        self.k = n_splits

    def split(self, data, target):
        kf = KFold(n_splits=self.k, shuffle=True)
        i = 1
        folders = []
        for train_index, test_index in kf.split(data):
            # print("folder", i, "length:", len(test_index))
            folders.append(test_index)
            # i += 1
        return folders


def seed_everything(seed=-1):
    if seed != -1:
        random.seed(seed)
        # os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print("Seed is set to ", seed)
    else:
        print("Seed is not set")


def process_data(data):
    data = pd.get_dummies(data, columns=['cp_time', 'cp_dose'])
    return data


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Running")
    # make the number not equals to 0 to make each training generate the same result.
    seed_everything(42)
    # Read Files
    # cp_... = cp_type + cp_time + cp_dose
    # train_features = 23814 * 876 = sig_id + cp_type + cp_... + features
    train_features = pd.read_csv('./input/lish-moa/train_features.csv')
    # train_targets_scored = 23814 * 207 = sig_id + 206 targets' scores
    train_targets_scored = pd.read_csv('./input/lish-moa/train_targets_scored.csv')
    # (Unknown use) 23814 * 403
    # train_targets_nonscored = pd.read_csv('./input/lish-moa/train_targets_nonscored.csv')
    # test_features = 3982 * 876 = sig_id + cp_type + cp_... + features
    test_features = pd.read_csv('./input/lish-moa/test_features.csv')
    # sample_submission = 3982 * 207 = sig_id + 206 targets
    sample_submission = pd.read_csv('./input/lish-moa/sample_submission.csv')

    # GENES = 772 names (g-0~711)
    GENES = [col for col in train_features.columns if col.startswith('g-')]
    # CELLS = 100 names (c-0~99)
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    # GENES
    n_comp = 50
    # data = 27796 * 772(only genes); 27796 = 23814(train) + 3982(test)
    data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
    # data2 = 27796 * 50, maybe chose 50 genes?
    data2 = (PCA(n_components=n_comp, random_state=42).fit_transform(data[GENES]))
    # re-divide train and test, but now, each has only 50 features.
    # train2 = 23814 * 50
    train2 = data2[:train_features.shape[0]]
    # test2 = 3982 * 50
    test2 = data2[-test_features.shape[0]:]
    # print(test2)
    train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])
    test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])
    # drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
    train_features = pd.concat((train_features, train2), axis=1)
    test_features = pd.concat((test_features, test2), axis=1)

    # CELLS
    n_comp = 15
    data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
    data2 = (PCA(n_components=n_comp, random_state=42).fit_transform(data[CELLS]))
    train2 = data2[:train_features.shape[0]]
    test2 = data2[-test_features.shape[0]:]
    train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])
    test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])
    # drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
    train_features = pd.concat((train_features, train2), axis=1)
    test_features = pd.concat((test_features, test2), axis=1)

    train_features = train_features.drop(GENES + CELLS, axis=1)
    test_features = test_features.drop(GENES + CELLS, axis=1)

    # print(test_features.head())

    from sklearn.feature_selection import VarianceThreshold

    var_thresh = VarianceThreshold(threshold=0.5)
    data = train_features.append(test_features)
    data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])

    train_features_transformed = data_transformed[: train_features.shape[0]]
    test_features_transformed = data_transformed[-test_features.shape[0]:]

    train_features = pd.DataFrame(train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 4),
                                  columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose'])
    train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)
    test_features = pd.DataFrame(test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 4),
                                 columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose'])
    # test features = 3982 * 931
    test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)

    # print(test_features.head())
    # print(train_features.shape)

    # train = id + cp_type + cp_... + features + scores of each item (targets)
    # 23814 * 1082/1137
    train = train_features.merge(train_targets_scored, on='sig_id')
    print(train.shape)
    # delete all rows with cp_type = ctl_vehicle and regenerate index (0~21947)
    # train = 21948 * 1082
    train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    # do the same thing to test_features
    # test = 3624 * 876
    test = test_features[test_features['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)

    # scores of data points which cp_type != ctl_vehicle
    # target = 21948 * 207 (id + 206 targets)
    target = train[train_targets_scored.columns]
    # train = 21948 * 1081 = id + cp_time + cp_dose + 872 features + 206 targets' scores
    train = train.drop('cp_type', axis=1)
    # test = 3624 * 875 (train - 206 targets)
    test = test.drop('cp_type', axis=1)
    # Get features' names about gene and cells (only names, not include any features' value)
    # That is, divided features into GENES and CELLS (ignore cp_time and cp_dose. They will be managed later)
    # target_cols = 1*206, a list which contains 206 columns'/targets' names in target except sig_id
    # .columns chang it to a special list. .values change it to a numpy array/matrix = 1*206
    # .tolist() change it to a general list.
    target_cols = target.drop('sig_id', axis=1).columns.values.tolist()
    # make a copy of train as folds 21948 * 1081 = id + cp_time + cp_dose + 872 features + 206 targets
    folds = train.copy()
    msfk = MultilabelStratifiedKFold(5)
    # split train data into 5 folders/groups
    results = msfk.split(folds, target)
    for i in range(len(results)):
        for k in range(len(results[i])):
            # assign folders number to each row of data (i.e. data point)
            folds.loc[results[i][k], 'kfold'] = int(i)
    # change value type to integer. Now folds = 21948 * 1082 = id + cp_time + cp_dose + 872 features + 206 targets
    # + kfold (the last column)
    folds['kfold'] = folds['kfold'].astype(int)
    # rebuild cp_time = {24,72,48} and cp_does = {'D1','D2'}
    # cp_time = cp_time_24 + cp_time_48 + cp_time_72
    # cp_dose = cp_dose_D1 + cp_dose_D2
    # new fold = 21948 * 1085 = id + 872 features + 206 targets' scores + kfold + 3*cp_time + 2*cp_dose
    temp_fold = process_data(folds)
    # First filter should generate a list with length 879 which contains the names of id + 872 features + kfold +
    # 3*cp_time + 2*cp_dose
    feature_cols = [c for c in temp_fold.columns if c not in target_cols]
    # Second filter should reduce the length of list to 877 by removing id and kfold
    # Hence, total features = 872 initial features + 3 cp_time + 2 cp_dose = 877 features
    # Finally, feature_cols = 877 features' names
    feature_cols = [c for c in feature_cols if c not in ['kfold', 'sig_id']]
    num_features = len(feature_cols)  # = 877
    num_targets = len(target_cols)  # = 206
    hidden_size = 1024


# Below is Single Fold Training Part
def run_training(fold, seed):
    seed_everything(seed)
    # Dummy cp_time and cp_dose. train_ = 21948 * 1085 (folds = 21948 * )
    train = process_data(folds)
    # Dummy cp_time and cp_dose. test_ = 3624 * 878 (test = 3624 * 875)
    test_ = process_data(test)
    # Get the index of all samples that not belong to folder_number = fold for training. About 17558 samples
    trn_idx = train[train['kfold'] != fold].index
    # Get the index of all samples that belong to folder_number = fold for validation. About 4390 samples
    val_idx = train[train['kfold'] == fold].index
    # divide train data into training part (4 folds) and validation part (1 fold) and reindex them.
    # Their columns = id + 877 features + 206 targets + kfold = 1085 columns
    train_df = train[train['kfold'] != fold].reset_index(drop=True)
    valid_df = train[train['kfold'] == fold].reset_index(drop=True)
    # Get each parts' data and targets
    # x_train ~= 17558 * 877, y_train ~= 17558 * 206
    x_train, y_train = train_df[feature_cols].values, train_df[target_cols].values
    # x_valid ~= 4390 * 877, y_valid ~= 4390 * 206
    x_valid, y_valid = valid_df[feature_cols].values, valid_df[target_cols].values
    # Put data and targets into two dataset (train ad validation)
    train_dataset = Datasets.MoADataset(x_train, y_train)
    valid_dataset = Datasets.MoADataset(x_valid, y_valid)
    # Build data loaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # build module with 877 features, 206 targets, 1024 hidden size
    model = Datasets.Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    # run model on CPU/Cuda
    model.to(DEVICE)
    # create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    # create loss function
    loss_fn = nn.BCEWithLogitsLoss()
    # set stop steps
    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0
    # train = 21948 * 1085 (only in this function)
    # target = 21948 * 207 (id + 206 targets)
    # oof = 21948 * 206, each element is 0
    oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
    # Set best loss that can be overwritten by any loss value
    best_loss = np.inf
    # training
    for epoch in range(EPOCHS):

        train_loss = Datasets.train_fn(model, optimizer, scheduler, loss_fn, trainloader, DEVICE)
        print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")
        valid_loss, valid_preds = Datasets.valid_fn(model, loss_fn, validloader, DEVICE)
        print(f"FOLD: {fold}, EPOCH: {epoch}, valid_loss: {valid_loss}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            oof[val_idx] = valid_preds
            torch.save(model.state_dict(), f"FOLD{fold}_.pth")
        elif EARLY_STOP:
            early_step += 1
            if early_step >= early_stopping_steps:
                break
    # assign the valid samples (get from val_idx) the best valid prediction (from valid_preds)
    # oof = 21948 * 206, only 4390 rows has non-zero (maybe zero) values on all columns,
    # the other rows have zero value on each column

    # Prediction Part
    # Get features' values from test data (from line 131: 3624 * 878): 3624*877,remove the id column
    x_test = test_[feature_cols].values
    # Build dataset
    testdataset = Datasets.TestDataset(x_test)
    # Build dataloader
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    # Build model (num_features = 877, num_targets = 206, hidden_size = 1024)
    model = Datasets.Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )

    model.load_state_dict(torch.load(f"FOLD{fold}_.pth"))
    model.to(DEVICE)

    # Unknown use
    predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
    # Get predictions = 3624 * 206
    predictions = Datasets.inference_fn(model, testloader, DEVICE)

    return oof, predictions


def run_k_fold(NFOLDS, seed):
    # Initialize oof (21948 * 206) and predictions (3624 * 206)
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))

    for fold in range(NFOLDS):
        oof_, pred_ = run_training(fold, seed)
        # Get the average prediction of test data
        predictions += pred_ / NFOLDS
        # Assign values to the rows which index is in first "fold"(1~5) folders
        oof += oof_
    # when the loop is finished all rows in oof should have value
    return oof, predictions




# Fix the random
SEED = [0, 1, 2, 3, 4, 5]
# Initialize
oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

for seed in SEED:
    print("Turn: seed =", seed)
    oof_, predictions_ = run_k_fold(NFOLDS, seed)
    # oof += oof_/5, get totally 5 oof which each row is assigned value, then get their average.
    oof += oof_
    # Get average value
    predictions += predictions_
    # I add this break because each seed cost about 20 minutes
    # break

# Before: train = 21984 * 1081, test = 3624 * 875 (id + cp_time + cp_dose + 872 gene & cell features)
# After: train = 21984 * 1081 (not 0 but close to 0 for some column)
# test = 3624 * 1081 (add additional 206 columns for targets)
train[target_cols] = oof
test[target_cols] = predictions
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# After drop target_cols, the train_targets_scored has only one line: sig_id
# By merging, the valid_result get 206 targets' columns back with oof values (valid_preds of all rows)
# NaN is replaced with 0
# valid_results = 23814 * 207 (because there are some rows which cp_type is ctl_vehicle)
valid_results = train_targets_scored.drop(columns=target_cols).merge(train[['sig_id'] + target_cols], on='sig_id',
                                                                     how='left').fillna(0)
# Get true targets' values (numpy matrix)
y_true = train_targets_scored[target_cols].values
# Get prediction values (numpy matrix)
y_pred = valid_results[target_cols].values

score = 0
# Get log loss of each column, then count average value
# target = 21948 * 207 (Why not use 206?) 207 = sig_id + 206 features
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / target.shape[1]
print("CV log_loss: ", score)
# sub = 3982 * 207
sub = sample_submission.drop(columns=target_cols).merge(test[['sig_id'] + target_cols], on='sig_id', how='left').fillna(
    0)
print("sub: ", sub.shape)
sub.to_csv('submission.csv', index=False)
