from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor

import argparse
import pathlib
import meta_data
import pandas as pd
import numpy as np

from meta_data import MetaData


def normalized_mse(y_pred, y_true):
    # """Normalized mean squared error regression loss
    # Parameters
    # ----------
    # y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
    #     Ground truth (correct) target values.
    # y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
    #     Estimated target values.
    # Returns
    # -------
    # loss : float or ndarray of floats
    #     A non-negative floating point value (the best value is 0.0), or an
    #     array of floating point values, one for each individual target.
    # """
    
    # mse = np.mean_squared_error(y_true, y_pred)
    mse = np.square(np.subtract(y_true, y_pred)).mean()
    means = np.ones_like(y_true) * y_true.mean()
    # norm = np.mean_squared_error(means, y_true)
    norm = np.square(np.subtract(means, y_true)).mean()
    return mse / norm

class MetaStream():
    """
    meta-learning recommendation algorithm for concept shift or drift in data streams.
    """


    def __init__(self, meta_learner, learners, train_window_size, sel_window_size, meta_features_train, meta_features_train_names): 
        
        # meta-learner
        self.meta_learner = meta_learner

        # regression models
        self.learners = learners

        # meta-featuers
        self.meta_features_train = meta_features_train
        self.meta_features_train_names = meta_features_train_names

        self.train_window_size = train_window_size
        self.sel_window_size = sel_window_size

        self.meta_table = pd.DataFrame(columns=self.meta_features_train_names + ['regressor'])

    def add_regressor(self, regressor, idx):
        self.meta_table.loc[idx, 'regressor'] = regressor

    def generate_meta_features_train(self, X, y):
        temp_dict = {}
        for i, (key, value) in enumerate(self.meta_features_train.items()):
            if len(value) == 1:
                temp_dict.update({self.meta_features_train_names[i] : key(key(X))})
            else:
                key(X, y)

        self.meta_table = self.meta_table.append(temp_dict, ignore_index=True)

    def base_fit(self, X, y):
        """
        base-learning needs to be run before the initiation of the meta-learner.
        
        generates:
            - generate the base data for the meta-learner
        
        parameters:
        ----------
            X : training data

            y : training labels(target values)

        returns:
        ----------
            None
        """


        [learner.fit(X, y) for learner in self.learners]


    def base_predict(self, X):
        """
        base fit needs to be run before calling this function.

        predict class value for X.

        parameters:
        ----------
            X : test data

        returns:
        ----------
            y : predicted labels(target values)
        """

        return [learner.predict(X) for learner in self.learners]

    def initial_fit(self, X, y):
        """
        parameters:
        ----------
            X : 

            y : 

        returns:
        ----------
            None
        """

        self.meta_learner.fit(X, y)


    def predict(self, X):
        """
        predict the recommended learner for X.

        parameters:
        ----------
            X : 

        returns:
        ----------
            y : 
        """

        return self.meta_learner.predict(X)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', type=pathlib.Path)
    parser.add_argument('-training_window_size', default=100, type=int, nargs=1)
    parser.add_argument('-selection_window_size', default=10, type=int, nargs=1)
    parser.add_argument('-window_size', default=100, type=int, nargs=1)
    parser.add_argument('-gamma_size', default=100, type=int, nargs=1)
    parser.add_argument('-initial_size', default=100, type=int, nargs=1)

    train_window_size = parser.parse_args().training_window_size
    sel_window_size = parser.parse_args().selection_window_size
    window_size = parser.parse_args().window_size
    gamma_size = parser.parse_args().gamma_size
    initial_size = parser.parse_args().initial_size

    print("training window size: ", train_window_size)
    print("selection window size: ", sel_window_size)
    print("window size: ", window_size)
    print("gamma size: ", gamma_size)
    print("initial size: ", initial_size)


    df = pd.read_csv(parser.parse_args().datapath)
    df['class'] = (df['class'] == "UP").astype(int)


    # NOTE: list of regression algorithms
    models =    [
                SVR(),
                RandomForestRegressor(random_state=42),
                GaussianProcessRegressor(random_state=42),
                LinearRegression(),
                Lasso(),
                Ridge(),
                GradientBoostingRegressor(random_state=42)
                ]

    # NOTE: meta-learner
    meta_learner = SGDClassifier()
    
    # NOTE: meta-features
    # meta_features_train = [np.mean, np.var, np.min, np.max, np.median]
    meta_features_train = {np.mean : ['attribute'], np.var : ['attribute'], np.min : ['attribute'], np.max : ['attribute'], np.median : ['attribute']}
    meta_features_train_names = ['mean', 'var', 'minimum', 'maximum', 'median']

    # NOTE: creates meta object
    metas = MetaStream(meta_learner, models, train_window_size, sel_window_size, meta_features_train, meta_features_train_names)


    for idx in range(initial_size):
        train = df.iloc[idx * gamma_size : idx * gamma_size + window_size]
        sel = df.iloc[idx * gamma_size + window_size : (idx + 1) * gamma_size + window_size]
        
        X_train, y_train =  train.drop('nswdemand', axis=1), train['nswdemand']
        X_sel, y_sel =  sel.drop('nswdemand', axis=1), train['nswdemand']

        # generate meta-features across entire X_train
        metas.generate_meta_features_train(X_train, y_train)

        # fit the regression models
        metas.base_fit(X_train, y_train)

        preds = metas.base_predict(X_sel)
        scores = [normalized_mse(pred, y_sel) for pred in preds]
        max_score = np.argmax(scores)

        # add best performing regression model to meta-data table
        metas.add_regressor(max_score, idx)



    print(metas.meta_table.describe())