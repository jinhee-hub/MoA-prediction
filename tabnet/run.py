##Implemented this and other files in this directory by Bo Peng

from dataLoder import data
from configuration import config
from utils import *
import numpy as np
import torch
import torch.optim
import torch.nn
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from model import TabNetRegressor
import pdb

if __name__ == '__main__':
    #fix seed
    setSeeds(0)

    path = '../input/lish-moa/'
    dataLoader = data(path)

    cfg = config(dataLoader)

    X_test = np.concatenate([dataLoader.catTest, dataLoader.numTest], axis=1)

    if cfg.strategy == 'KFOLD':
        oof_preds_all   = []
        oof_targets_all = []
        scores_all     = []
        scores_auc_all  = []
        preds_test      = []

        for seed in range(cfg.numEnsembling):
            print(" SEED : ", seed)
        mskf = MultilabelStratifiedKFold(n_splits=cfg.SPLITS, random_state=cfg.seed+seed, shuffle=True)

        oof_preds   = []
        oof_targets = []
        scores      = []
        scores_auc  = []

        p = []
        for j, (train_idx, val_idx) in enumerate(mskf.split(np.zeros(len(dataLoader.catTrain)), dataLoader.targetsTrain)):
            print("FOLDS : ", j)

            #model
            X_train, y_train = torch.as_tensor(np.concatenate((dataLoader.catTrain[train_idx], dataLoader.numTrain[train_idx]), axis=1)), torch.as_tensor(dataLoader.targetsTrain[train_idx])

            X_val, y_val = torch.as_tensor(np.concatenate((dataLoader.catTrain[val_idx], dataLoader.numTrain[val_idx]), axis=1)), torch.as_tensor(dataLoader.targetsTrain[val_idx])

            model = TabNetRegressor(n_d=48, n_a=48, n_steps=1, gamma=1.3, lambda_sparse=0, cat_dims=cfg.catDims, cat_emb_dim=cfg.catEmbDim, cat_idxs=cfg.catsIdx, optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=1e-2), mask_type='entmax', device_name=cfg.device, scheduler_params=dict(milestones=[ 50,100,150], gamma=0.9), scheduler_fn=torch.optim.lr_scheduler.MultiStepLR)

            name = cfg.saveName + f"_fold{j}_{seed}"
            #model.load_model(name)
            model.fit(X_train = X_train, y_train = y_train, X_valid = X_val, y_valid = y_val, patience=20, batch_size=1024, virtual_batch_size=128, drop_last=False, loss_fn=torch.nn.functional.binary_cross_entropy_with_logits)
            preds = model.predict(X_val)
            preds = torch.sigmoid(torch.as_tensor(preds)).detach().cpu().numpy()

            y_val_numpy = y_val.detach().cpu().numpy()
            score = logLossMulti(y_val_numpy, preds)

            temp = model.predict(X_test)
            p.append(torch.sigmoid(torch.as_tensor(temp)).detach().cpu().numpy())
            ## save oof to compute the CV later
            oof_preds.append(preds)
            oof_targets.append(y_val_numpy)
            scores.append(score)
            scores_auc.append(AUC(y_val_numpy,preds))
            print(f"validation fold {j} : {score}")
        p = np.stack(p)
        preds_test.append(p)
        oof_preds_all.append(np.concatenate(oof_preds))
        oof_targets_all.append(np.concatenate(oof_targets))
        scores_all.append(np.array(scores))
        scores_auc_all.append(np.array(scores_auc))

        preds_test = np.stack(preds_test)

        if cfg.strategy == "KFOLD":

            for i in range(cfg.numEnsembling): 
                print("CV score fold : ", logLossMulti(oof_targets_all[i], oof_preds_all[i]))
                print("auc mean : ", sum(scores_auc_all[i])/len(scores_auc_all[i]))
