import numpy as np
from sklearn.metrics import roc_auc_score
from dataLoder import data
import torch


class config(object):
    def __init__(self, dataloader):
        self.numClass = dataloader.targetsTrain.shape[1]
        self.verbose = False
        #
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.SPLITS= 7
        self.EPOCHS = 200
        self.numEnsembling = 1
        self.seed = 0

        self.catEmbDim = [1] * dataloader.catTrain.shape[1]
        self.catsIdx   = list(range(dataloader.catTrain.shape[1]))
        self.catDims   = [len(np.unique(dataloader.catTrain[:,i])) for i in self.catsIdx]
        #self.numNum is the number of nummerical features in training
        self.numNum= dataloader.numTrain.shape[1]
        self.saveName = "models/first_test"

        self.strategy = "KFOLD"

#test
if __name__ == '__main__':
    dataloder = data('../input/lish-moa/') 
    cfg = config(dataloder)
    pdb.set_trace()

print('nb')  
