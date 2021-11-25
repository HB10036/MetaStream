from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import cohen_kappa_score, mean_squared_error, classification_report, accuracy_score, make_scorer
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced

import argparse
import pathlib
import meta_data
import pandas as pd
import numpy as np

from meta_data import MetaData
from util import normalized_mse

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

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

        # self.meta_table = pd.DataFrame(columns=self.meta_features_train_names + ['regressor'])
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  ['regressor']
        self.meta_table = pd.DataFrame(columns=[str(0), str(1), str(2), str(3), str(4), str(5), str(6), str(7), str(8), str(9), str(10), str(11), str(12)] + ['regressor'])

    def add_regressor(self, regressor, idx):
        self.meta_table.loc[idx, 'regressor'] = regressor

    def generate_meta_features_train(self, X, y):
        ecol = importr("ECoL")
        temp = {}
        with localconverter(ro.default_converter + pandas2ri.converter):

            rfeatures = ecol.complexity(X, y, summary=["mean"])
            for i, value in enumerate(rfeatures):
                temp.update({str(i) : value})

        self.meta_table = self.meta_table.append(temp, ignore_index=True)

        # temp_dict = {}
        # for i, (key, value) in enumerate(self.meta_features_train.items()):
        #     if len(value) == 1:
        #         temp_dict.update({self.meta_features_train_names[i] : key(key(X))})
        #     else:
        #         key(X, y)

        # self.meta_table = self.meta_table.append(temp_dict, ignore_index=True)

    def get_meta_features(self, X, y):
        ecol = importr("ECoL")
        temp = {}
        with localconverter(ro.default_converter + pandas2ri.converter):

            rfeatures = ecol.complexity(X, y, summary=["mean"])
            for i, value in enumerate(rfeatures):
                temp.update({str(i) : value})

        return temp
        # temp_dict = []
        # for i, (key, value) in enumerate(self.meta_features_train.items()):
        #     if len(value) == 1:
        #         temp_dict.append(key(key(X)))
        #     else:
        #         key(X, y)

        # return temp_dict

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


    # initializing base meta-data 
    for idx in range(initial_size):
        train = df.iloc[idx * gamma_size : idx * gamma_size + window_size]
        sel = df.iloc[idx * gamma_size + window_size : (idx + 1) * gamma_size + window_size]
        
        X_train, y_train =  train.drop('nswdemand', axis=1), train['nswdemand']
        X_sel, y_sel =  sel.drop('nswdemand', axis=1), sel['nswdemand']

        # generate meta-features across entire X_train
        metas.generate_meta_features_train(X_train, y_train)

        # fit the regression models
        metas.base_fit(X_train, y_train)

        preds = metas.base_predict(X_sel)
        scores = [normalized_mse(pred, y_sel) for pred in preds]
        max_score = np.argmin(scores)

        # add best performing regression model to meta-data table
        metas.add_regressor(max_score, idx)


    print(metas.meta_table.info())
    print(metas.meta_table.head())

    mxtrain, mxtest, mytrain, mytest = train_test_split(metas.meta_table.drop(['regressor'], axis=1), metas.meta_table['regressor'], random_state=42)

    metas.initial_fit(mxtrain, mytrain)
    myhattest = metas.predict(mxtest)
    print("Kappa: ", cohen_kappa_score(mytest, myhattest))
    print("GMean: ", geometric_mean_score(mytest, myhattest))
    print("Accuracy: ", accuracy_score(mytest, myhattest))
    print(classification_report(mytest, myhattest))
    print(classification_report_imbalanced(mytest, myhattest))

    m_recommended = []
    m_best = []

    score_recommended = []
    score_default = []

    small_data = 5000
    until_data = min(initial_size + small_data, int((df.shape[0] - window_size) / gamma_size))
    # print(small_data, until_data)

    count = 0

    for idx in range(initial_size, until_data):
        # print(idx)
        train = df.iloc[idx * gamma_size : idx * gamma_size + window_size]
        sel = df.iloc[idx * gamma_size + window_size : (idx + 1) * gamma_size + window_size]
        
        X_train, y_train =  train.drop('nswdemand', axis=1), train['nswdemand']
        X_sel, y_sel =  sel.drop('nswdemand', axis=1), sel['nswdemand']

        mfs = metas.get_meta_features(X_train, y_train)
        predicted_model = int(metas.predict(np.array(list(mfs.values())).reshape(1, -1))[0])
        m_recommended.append(predicted_model)
        
        # print(metas.learners[predicted_model])
        # print(metas.learners[predicted_model].fit(X_train, y_train).predict(X_sel))
        # print(list(y_sel))
        score1 = normalized_mse(list(y_sel), metas.learners[predicted_model].fit(X_train, y_train).predict(X_sel))
        score_recommended.append(score1)
        # print(score1)


        # fit the regression models
        metas.base_fit(X_train, y_train)

        preds = metas.base_predict(X_sel)
        scores = [normalized_mse(pred, y_sel) for pred in preds]
        max_score = np.argmin(scores)
        # print(scores[max_score])
        metas.meta_table = metas.meta_table.append(mfs, ignore_index=True)
        metas.add_regressor(max_score, idx)
        metas.initial_fit(metas.meta_table.drop(['regressor'], axis=1)[-100:], metas.meta_table['regressor'][-100:])

        m_best.append(max_score)

        # need to still add meta data to meta data table
    print("Kappa: ", cohen_kappa_score(m_best, m_recommended))
    print("GMean: ", geometric_mean_score(m_best, m_recommended))
    print("Accuracy: ", accuracy_score(m_best, m_recommended))
    print(classification_report(m_best, m_recommended))
    print(classification_report_imbalanced(m_best, m_recommended))

    print("Mean score Recommended {:.2f}+-{:.2f}".format(np.mean(score_recommended), np.std(score_recommended)))


