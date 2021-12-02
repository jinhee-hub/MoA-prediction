import numpy as np
import pandas as pd
import os
import pdb

class data(object):
    def __init__(self, path):
         
        for dirName, _, fileNames in os.walk(path):
            for fileName in fileNames:
                print(os.path.join(dirName, fileName))

        self.train            = pd.read_csv(path+'train_features.csv')
        self.trainTarScore    = pd.read_csv(path+'train_targets_scored.csv')
        self.trainNonTarScore = pd.read_csv(path+'train_targets_nonscored.csv')
        self.testFeatures     = pd.read_csv(path+'test_features.csv')
        self.submission       = pd.read_csv(path+'sample_submission.csv')

        self.removeVehicle = True

        if self.removeVehicle:
            self.trainFeatures = self.train.loc[self.train['cp_type']=='trt_cp'].reset_index(drop=True)
            self.trainTarScore = self.trainTarScore.loc[self.train['cp_type']=='trt_cp'].reset_index(drop=True)
            self.trainNonTarScore = self.trainNonTarScore.loc[self.train['cp_type']=='trt_cp'].reset_index(drop=True)
        else:
            self.trainFeatures = self.train

        scores, scoresRatio        = self.getRatioLabels(self.trainTarScore)
        nonScores, nonScoresRation = self.getRatioLabels(self.trainNonTarScore)

        colFeatures = list(self.trainFeatures.columns)[1:]
        self.catTrain, self.catTest, self.numTrain, self.numTest = self.transform(self.trainFeatures, self.testFeatures, colFeatures, normalize=False)

        self.targetsTrain   = self.trainTarScore[scores].values.astype(np.float32)
        self.targetsOpTrain = self.trainNonTarScore[nonScores].values.astype(np.float32)

    def getRatioLabels(self, df):
        #normalization, remove labels that all 0 or 1
        columns = list(df.columns)
        #pop the index col
        columns.pop(0)

        ratios = []
        remove = []

        for c in columns:
            counts = df[c].value_counts()
            if len(counts) != 1:
                ratios.append(counts[0]/counts[1])
            else:
                remove.append(c)

        print("remove %d columns" % len(remove))

        for r in remove:
            columns.remove(r)
        
        return columns, np.array(ratios)

    def transform(self, train, test, col, normalize = True):
        #normalization
        #use these value for now, will change later
        max_ = 10
        min_ = -10

        #dictionary of dictionary to convert category features to one-hot embedding
        mapping = {"cp_type":{"trt_cp": 0, "ctl_vehicle":1},
               "cp_time":{48:0, 72:1, 24:2},
               "cp_dose":{"D1":0, "D2":1}}

        if self.removeVehicle:
            categoriesTrain = np.stack([ train[c].apply(lambda x: mapping[c][x]).values for c in col[1:3]], axis=1)
            categoriesTest  = np.stack([ test[c].apply(lambda x: mapping[c][x]).values for c in col[1:3]], axis=1)

        else:
            categoriesTrain = np.stack([ train[c].apply(lambda x: mapping[c][x]).values for c in col[:3]], axis=1)
            categoriesTest  = np.stack([ test[c].apply(lambda x: mapping[c][x]).values for c in col[:3]], axis=1)

        #filter out the first 3
        numericalTrain = train[col[3:]].values
        numericalTest  = test[col[3:]].values

        if normalize:
            numericalTrain = (numericalTrain-min_)/(max_ - min_)
            numericalTest  = (numericalTest-min_)/(max_-min_)

        return categoriesTrain, categoriesTest, numericalTrain, numericalTest
            

#unit test
if __name__ == "__main__":
    dataLoader = data('lish_moa/')
    scores, scoresRatio        = dataLoader.getRatioLabels(dataLoader.trainTarScore)
    nonScores, nonScoresRation = dataLoader.getRatioLabels(dataLoader.trainNonTarScore)

    colFeatures = list(trainFeatures.columns)[1:]
    catTrain, catTest, numTrain, numTest = dataLoader.transform(trainFeatures, testFeatures, colFeatures, normalize=False)

    targetsTrain   = dataLoader.trainTarScore[scores].values.astype(np.float32)
    targetsOpTrain = dataLoader.trainNonTarScore[nonScores].values.astype(np.float32)


