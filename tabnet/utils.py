import torch
import numpy as np
import torch.nn
import os
import random
import pdb
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

def setSeeds(seedValue):
    #fix the seed of all the random generator
    random.seed(seedValue)
    np.random.seed(seedValue)
    torch.manual_seed(seedValue)
    os.environ['PYTHONHASHSEED'] = str(seedValue)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seedValue)
        torch.cuda.manual_seed_all(seedValue)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def inference(model, X, verbose = True):
    with torch.no_grad():
        y_preds = model.predict(X)
        y_preds = torch.sigmoid(torch.as_tensor(y_preds)).numpy()

    return y_preds

def logLossScore(predicted, actual, eps=1e-12):
    p1 = actual * np.log(predicted+eps)
    p0 = (1-actual) * np.log(1-predicted+eps)

    loss = p1 + p0

    return -loss.mean()

def AUC(target, pred):
    M   = target.shape[1]
    res = np.zeros(M)
    for i in range(M):
        try:
            res[i] = roc_auc_score(target[:,i], pred[:,i])
        except:
            pass
    return res.mean()

def logLossMulti(y_true, y_pred):
    M = y_true.shape[1]
    results = np.zeros(M)
    for i in range(M):
        results[i] = logLossScore(y_true[:,i], y_pred[:,i])
    return results.mean()
        
