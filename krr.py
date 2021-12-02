## Jinhee Lee wrote this code
# this code run in the local environment

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import log_loss
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# data preprocessing. We change all data as we easily treat
def preprocessing(data):  # data is all datas in .csv file (train or test_features)
    data = data.copy()

    # in cp_type column, it has only two parameters: trt_cp and ct1_vehicle.
    # in cp_dose column, it has only two parameters: D1 and D2
    # rename parameters in cp_type and cp_dose
    # transfer(map) trt_cp -> 0 and ct1_vehicle -> 1 in cp_type column
    data.loc[:, 'cp_type'] = data.loc[:, 'cp_type'].map({'trt_cp' : 0 , 'ctl_vehicle':1})
    # transfer D1 to 0 and D2 to 1
    data.loc[:, 'cp_dose'] = data.loc[:, 'cp_dose'].map({'D1': 0, 'D2':1})
    #print(data.loc[:, 'cp_type'])
    #print(data.loc[:, 'cp_dose'])
    # we don’t need sig_id, so delete it
    del data['sig_id']

    return data

# MOA offers the log loss function: to evaluate the accuracy of a solution
# in Evaluation tap in MOA kaggle, we can get an equation.
def logLoss(y, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1- 1e-10)
    loss = -np.mean(np.mean(y * np.log(y_pred) + (1-y)*np.log(1-y_pred), axis=1))

    return loss

# Get data from .csv files (MoA prediction offers all of dataset)
# The size of train_features.csv is large, need to download from URL:  https://www.kaggle.com/c/lish-moa/data
train_features = pd.read_csv('../input/lish-moa/train_features.csv')
# MoA also offers non-scored, but we only concern scored targets
train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
test_features = pd.read_csv('../input/lish-moa/test_features.csv')
# sample submission is a format what MoA want to submit
sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')

# if c!= 'sig_id' then c = 1st row's values
cols = [ c for c in sample_submission.columns.values if c != 'sig_id']
# there are genes and cells in features
genes=[col for col in train_features.columns if col.startswith('g-')]
cells=[col for col in train_features.columns if col.startswith('c-')]

# just make some parameters to 0 or 1
train=preprocessing(train_features)
test=preprocessing(test_features)
# also delete sig_id in train_targets
del train_targets['sig_id']
# only get cp_type=0 data in the original dataset.
# and re-index these data. Because some data are removed, then index may become like 0,1,3,7,10....
# then, we need to make these index like 0,1,2,3,4,5...
train_targets=train_targets.loc[train['cp_type']==0].reset_index(drop=True)
train = train.loc[train['cp_type'] == 0].reset_index(drop=True)

### Kernel Logistic Regression
### this baseline uses RBF kernel -> use Nystoem approximate
tt = train_targets.copy()  # train_target data
ss=sample_submission.copy()   # sample submission
# zero setting
# make all data in each columns in range[:, train_targets.columns] as 0
ss.loc[:, train_targets.columns] = 0
tt.loc[:, train_targets.columns] = 0

# KFold means Cross Validation between train and test
# KFold is used to verify model's performance. In this case use KFold to the multilabel data
sc=[]
k = 7
for n, (i, j) in enumerate(MultilabelStratifiedKFold(n_splits = k, random_state=0, shuffle = True).split(train_targets, train_targets)):

    # 1. split train dataset as n splits : dataset/n  : this separates train_dataset as train(i) and test(j)
    # 2. choose one splitted sub_dataset as test(j) and N-1 sub_datasets are train(i)
    # compare data in 'i's and 'j'.
    x_tr, x_val = train.values[i][:, ], train.values[j][:, ]
    # do same as target dataset
    y_tr, y_val = train_targets.astype(float).values[i], train_targets.astype(float).values[j]
    x_tt=test.values[:, ]
    # use standardScaler: make all data in dataset as (all data - average of all data)/(standard derivation)
    # it makes variance as 1
    scaler=StandardScaler()
    x_tr=scaler.fit_transform(x_tr)  # make standardization of each data
    x_val=scaler.transform(x_val)
    x_tt=scaler.transform(x_tt)

    # use rbf kernel for Logistic regression
    model = KernelRidge(alpha = 100, kernel='rbf')
    model.fit(x_tr, y_tr)  # make model using data x_tr and y_tr

    # sample_submission and tt are already zero-setting before.
    # update submission as predicted one. model is used to predict
    ss.loc[:, train_targets.columns] += model.predict(x_tt)/k # this is for feature data
    fold_pred=model.predict(x_val)   # apply it for the target data
    tt.loc[j, train_targets.columns] += fold_pred # update predicted target in tt

    # get score using log loss function using this model.
    fold_score = logLoss(train_targets.loc[j], fold_pred)
    sc.append(fold_score)
    print(f'Kernel Ridge Regression: Fold {n}:', fold_score) # kernel ridge
    print(f'OOF Metric: {logLoss(train_targets.values, tt.values)}')
print(f'Average score: ', np.mean(sc)) # to get Average

### Platt Scaling
tt_new = tt[cols].values
ss_new = ss[cols].values
ttc = train_targets.copy()
ss.loc[:, train_targets.columns] = 0
ttc.loc[:, train_targets.columns] = 0

for tar in tqdm(range(train_targets.shape[1])):
    targets = train_targets.values[:, tar]

    if targets.sum() >= k:
        for n, (i,j) in enumerate(StratifiedKFold(n_splits=k, random_state=0, shuffle=True).split(targets, targets)):
            x_tr, x_val = tt_new[i, tar].reshape(-1,1), tt_new[j, tar].reshape(-1,1)
            y_tr, y_val = targets[i], targets[j]
            # uses Logistic regression
            model=LogisticRegression(penalty='none', max_iter = 1000)
            model.fit(x_tr, y_tr)
            ss.loc[:, train_targets.columns[tar]]+=model.predict_proba(ss_new[:, tar].reshape(-1, 1))[:, 1]/k
            ttc.loc[j, train_targets.columns[tar]] += model.predict_proba(x_val)[:,1]
            score=log_loss(train_targets.loc[:, train_targets.columns[tar]].values, ttc.loc[:, train_targets.columns[tar]].values)

print(f'Logistic Regression OOF Metric: {logLoss(train_targets.values, ttc.values)}') # logistic regression


