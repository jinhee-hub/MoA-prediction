import sys
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
sys.path.append('../input/iterative-stratification/iterative-stratification-master')
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow_addons as tfa
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
#from tqdm.notebook import tqdm
import tqdm
from typing import Tuple, List, Callable, Any
from sklearn.utils import check_random_state  # type: ignore
from utils import preprocess, pca, var_threshold, label_smoothing

train_features = pd.read_csv('input/lish-moa/train_features.csv')
train_targets = pd.read_csv('input/lish-moa/train_targets_scored.csv')
test_features = pd.read_csv('input/lish-moa/test_features.csv')

ss = pd.read_csv('input/lish-moa/sample_submission.csv')

train = preprocess(train_features)
test = preprocess(test_features)

#PCA and variance thresholding
##train, test = pca(train, test)
##train, test = var_threshold(train, test)

del train_targets['sig_id']                                                     #delete sig_id

train_targets_raw = train_targets.loc[train['cp_type'] == 0].reset_index(drop=True) #Take only the trp_cp data and delete rest
train = train.loc[train['cp_type'] == 0].reset_index(drop=True)                 #Take only the trp_cp data and delete rest
print(train.shape[1])

train_targets = label_smoothing(train_targets_raw, 0.01)

def create_model(num_columns):
    model = tf.keras.Sequential([ #plain stack of layers(1 input 1 output)  //deep learning model - why it works. fully connencted interationcs
        tf.keras.layers.Input(num_columns),
        tf.keras.layers.BatchNormalization(), #normalizing input and scaling and shifting it/ stabilize learning process and reduce # of epochs
        tf.keras.layers.Dropout(0.2), #for preventing overfitting
        #tfa.layers.WeightNormalization(tf.keras.layers.Dense(4096, activation="relu")),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Dropout(0.5),
        tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation="relu")),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tfa.layers.WeightNormalization(tf.keras.layers.Dense(1024, activation="relu")),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        #tfa.layers.WeightNormalization(tf.keras.layers.Dense(512, activation="relu")),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Dropout(0.5),
        #tfa.layers.WeightNormalization(tf.keras.layers.Dense(256, activation="relu")),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Dropout(0.5),

        tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, activation="sigmoid"))
    ])
    model.compile(optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(), sync_period=10),
                  loss='binary_crossentropy',       #training environment - optimizer,(lookahead) loss function
                  )
    '''
    The optimizer iteratively updates two sets of weights: the search directions for weights are chosen by the inner optimizer, 
    while the "slow weights" are updated each k steps based on the directions of the "fast weights" and the two sets of weights are synchronized. 
    This method improves the learning stability and lowers the variance of its inner optimizer.
    '''
    return model

'''
#Permutation importance -> remove the feature that is not important
#from eli5
def iter_shuffled(X, columns_to_shuffle=None, pre_shuffle=False,
                  random_state=None): #shuffle columns
    """
        Return an iterator of X matrices which have one or more columns shuffled.
        After each iteration yielded matrix is mutated inplace, so
        if you want to use multiple of them at the same time, make copies.

        ``columns_to_shuffle`` is a sequence of column numbers to shuffle.
        By default, all columns are shuffled once, i.e. columns_to_shuffle
        is ``range(X.shape[1])``.

        If ``pre_shuffle`` is True, a copy of ``X`` is shuffled once, and then
        result takes shuffled columns from this copy. If it is False,
        columns are shuffled on fly. ``pre_shuffle = True`` can be faster
        if there is a lot of columns, or if columns are used multiple times.
        """

    rng = check_random_state(random_state)


    if columns_to_shuffle is None:
        columns_to_shuffle = range(X.shape[1])

    print('column to shuffle ', columns_to_shuffle)

    if pre_shuffle:
        X_shuffled = X.copy()
        rng.shuffle(X_shuffled)

    X_res = X.copy()
    for columns in tqdm(columns_to_shuffle, disable=True):
        if pre_shuffle:
            X_res[:, columns] = X_shuffled[:, columns]
        else:
            rng.shuffle(X_res[:, columns])
        yield X_res                             #when results iterates, use yield
        X_res[:, columns] = X[:, columns]       #return to original


def get_score_importances(
        score_func,  # type: Callable[[Any, Any], float] # _score function
        X,
        y,
        n_iter=5,  # type: int
        columns_to_shuffle=None,
        random_state=None
):
    # type: (...) -> Tuple[float, List[np.ndarray]]
    """
    Return ``(base_score, score_decreases)`` tuple with the base score and
    score decreases when a feature is not available.

    ``base_score`` is ``score_func(X, y)``; ``score_decreases``
    is a list of length ``n_iter`` with feature importance arrays
    (each array is of shape ``n_features``); feature importances are computed
    as score decrease when a feature is not available.

    ``n_iter`` iterations of the basic algorithm is done, each iteration
    starting from a different random seed.

    If you just want feature importances, you can take a mean of the result::

        import numpy as np
        from eli5.permutation_importance import get_score_importances

        base_score, score_decreases = get_score_importances(score_func, X, y)
        feature_importances = np.mean(score_decreases, axis=0)

    """
    rng = check_random_state(random_state)
    base_score = score_func(X, y) # your function with model inference and scoring// mean
    scores_decreases = []
    for i in range(n_iter):
        scores_shuffled = _get_scores_shufled(
            score_func, X, y, columns_to_shuffle=columns_to_shuffle,
            random_state=rng, base_score=base_score
        )
        scores_decreases.append(scores_shuffled)

    print('in get score importance: base score ', base_score, 'scores decreases ', scores_decreases)

    return base_score, scores_decreases


def _get_scores_shufled(score_func, X, y, base_score, columns_to_shuffle=None,
                        random_state=None):
    Xs = iter_shuffled(X, columns_to_shuffle, random_state=random_state)
    res = []
    for X_shuffled in Xs:
        res.append(-score_func(X_shuffled, y) + base_score)

    return res


def metric(y_true, y_pred):
    metrics = []
    for i in range(y_pred.shape[1]):
        if y_true[:, i].sum() > 1:
            metrics.append(log_loss(y_true[:, i], y_pred[:, i].astype(float))) #log_loss(y_true, y_pred) cross entropy loss (-logP(yt/yp)= -(ytlog(yp)+(1-yt)log(1-yp))
    return np.mean(metrics)


perm_imp = np.zeros(train.shape[1])
all_res = []
for seed in range(7):
        for n, (tr, te) in enumerate(KFold(n_splits=7, random_state=0, shuffle=True).split(train_targets)): #split using train targets #Kfold - divides by k and k-1 used for training
            print(f'Fold {n}')

            print('order: seed, n, (tr, te) ', seed, n, (tr, te)) #tr 18816, te 3136


            model = create_model(len(train.columns))
            checkpoint_path = f'repeat{seed}_Fold{n}.hdf5'
            reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min') #when stopped improving, reduce lr
            cb_checkpt = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True,
                                         save_weights_only=True, mode='min') #Callback to save the Keras model or model weights at some frequency.
            model.fit(train.values[tr],
                      train_targets.values[tr],
                      validation_data=(train.values[te], train_targets.values[te]),
                      epochs=30, batch_size=128,
                      callbacks=[reduce_lr_loss, cb_checkpt], verbose=2
                      ) #Trains the model for a fixed number of epochs (iterations on a dataset).

            print(checkpoint_path)
            model.load_weights(checkpoint_path) #Loads all layer weights, either from a TensorFlow or an HDF5 weight file.


            def _score(X, y):
                pred = model.predict(X)
                return metric(y, pred)


            base_score, local_imp = get_score_importances(_score, train.values[te], train_targets.values[te], n_iter=1,
                                                      random_state=0)           #local_imp = score decrease
            all_res.append(local_imp)
            perm_imp += np.mean(local_imp, axis=0)
            print('')

top_feats = np.argwhere(perm_imp < 0).flatten() #find location of the condition / and flattens [0,1],[2,3] into [0,1,2,3]
print(top_feats)
'''

def metric(y_true, y_pred):
    metrics = []
    for _target in train_targets.columns:
        metrics.append(log_loss(y_true.loc[:, _target], y_pred.loc[:, _target].astype(float), labels=[0, 1]))
    return np.mean(metrics)


N_STARTS = 1 
tf.random.set_seed(42)
###############################################
feats = []                                    # Should comment out this part when using permutation importance
for i in range(train.shape[1]):               #
    feats.append(i)                           #
###############################################

res = train_targets.copy() #train targets
print('before =0 ', res)
ss.loc[:, train_targets.columns] = 0
res.loc[:, train_targets.columns] = 0
print('ss.loc ', ss, 'res.loc ', res)

for seed in range(N_STARTS):
    for n, (tr, te) in enumerate(
            MultilabelStratifiedKFold(n_splits=7, random_state=seed, shuffle=True).split(train_targets, train_targets)):
        print(f'Fold {n}')
        print(f'seed {seed}')

        """Multilabel stratified K-Folds cross-validator
            Provides train/test indices to split multilabel data into train/test sets.
            This cross-validation object is a variation of KFold that returns
            stratified folds for multilabel data. The folds are made by preserving
            the percentage of samples for each label.
            Parameters
            ----------
            n_splits : int, default=3
                Number of folds. Must be at least 2.
            shuffle : boolean, optional
                Whether to shuffle each stratification of the data before splitting
                into batches.
            random_state : int, RandomState instance or None, optional, default=None
                If int, random_state is the seed used by the random number generator;
                If RandomState instance, random_state is the random number generator;
                If None, the random number generator is the RandomState instance used
                by `np.random`. Unlike StratifiedKFold that only uses random_state
                when ``shuffle`` == True, this multilabel implementation
                always uses the random_state since the iterative stratification
                algorithm breaks ties randomly.
            Examples
            --------
            >>> from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
            >>> import numpy as np
            >>> X = np.array([[1,2], [3,4], [1,2], [3,4], [1,2], [3,4], [1,2], [3,4]])
            >>> y = np.array([[0,0], [0,0], [0,1], [0,1], [1,1], [1,1], [1,0], [1,0]])
            >>> mskf = MultilabelStratifiedKFold(n_splits=2, random_state=0)
            >>> mskf.get_n_splits(X, y)
            2
            >>> print(mskf)  # doctest: +NORMALIZE_WHITESPACE
            MultilabelStratifiedKFold(n_splits=2, random_state=0, shuffle=False)
            >>> for train_index, test_index in mskf.split(X, y):
            ...    print("TRAIN:", train_index, "TEST:", test_index)
            ...    X_train, X_test = X[train_index], X[test_index]
            ...    y_train, y_test = y[train_index], y[test_index]
            TRAIN: [0 3 4 6] TEST: [1 2 5 7]
            TRAIN: [1 2 5 7] TEST: [0 3 4 6]
            Notes
            -----
            Train and test sizes may be slightly different in each fold.
            See also
            --------
            RepeatedMultilabelStratifiedKFold: Repeats Multilabel Stratified K-Fold
            n times.
            """
        """Split():
        Generate indices to split data into training and test set.
                Parameters
                ----------
                X : array-like, shape (n_samples, n_features)
                    Training data, where n_samples is the number of samples
                    and n_features is the number of features.
                    Note that providing ``y`` is sufficient to generate the splits and
                    hence ``np.zeros(n_samples)`` may be used as a placeholder for
                    ``X`` instead of actual training data.
                y : array-like, shape (n_samples, n_labels)
                    The target variable for supervised learning problems.
                    Multilabel stratification is done based on the y labels.
                groups : object
                    Always ignored, exists for compatibility.
                Returns
                -------
                train : ndarray
                    The training set indices for that split.
                test : ndarray
                    The testing set indices for that split.
                Notes
                -----
                Randomized CV splitters may return different results for each call of
                split. You can make the results identical by setting ``random_state``
                to an integer.
                """

        model = create_model(len(feats))    #top feats into feats for not using permutation importance
        checkpoint_path = f'repeat{seed}_Fold{n}.hdf5'
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4,
                                           mode='min')
        cb_checkpt = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True,
                                     save_weights_only=True, mode='min')
        model.fit(train.values[tr][:, feats], #from training data extract only the top feats indices
                  train_targets.values[tr],
                  validation_data=(train.values[te][:, feats], train_targets.values[te]),
                  epochs=30, batch_size=128,
                  callbacks=[reduce_lr_loss, cb_checkpt], verbose=2
                  )

        model.load_weights(checkpoint_path)
        test_predict = model.predict(test.values[:, feats])
        val_predict = model.predict(train.values[te][:, feats])

        ss.loc[:, train_targets.columns] += test_predict
        res.loc[te, train_targets.columns] += val_predict
        print('')

ss.loc[:, train_targets.columns] /= ((n + 1) * N_STARTS) #starts from zero and does not include the last element
res.loc[:, train_targets.columns] /= N_STARTS

print(f'OOF Metric: {metric(train_targets_raw, res)}')

ss.loc[test['cp_type'] == 1, train_targets.columns] = 0

ss.to_csv('submission.csv', index=False)

