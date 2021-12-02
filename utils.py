import pandas as pd
import numpy as np
import pdb
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

def preprocess(df):
    df = df.copy()
    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})  #cp_type map trt_cp to 0, ctl_vehicle to 1
    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1}) #cp_dose map cp_dose D1 to 0, D2 to 1
    del df['sig_id']                                                    #sig_id delete
    return df

# Zhenhao Lu wrote this part
def pca(train_features, test_features):
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

    return train_features, test_features

# Zhenhao Lu wrote this part
def var_threshold(train_features, test_features):
    var_thresh = VarianceThreshold(threshold=0.5)
    data = train_features.append(test_features)
    data_transformed = var_thresh.fit_transform(data.iloc[:, 3:])

    train_features_transformed = data_transformed[: train_features.shape[0]]
    test_features_transformed = data_transformed[-test_features.shape[0]:]

    train_features = pd.DataFrame(train_features[['cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 3),
                                  columns=['cp_type', 'cp_time', 'cp_dose'])
    train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)
    test_features = pd.DataFrame(test_features[['cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 3),
                                 columns=['cp_type', 'cp_time', 'cp_dose'])
    test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)

    return train_features, test_features

def label_smoothing(target, alf):
    target = (1-alf) * target + alf/206

    return target
