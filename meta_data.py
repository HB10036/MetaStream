from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor

import numpy as np 
import scipy

class MetaData():

    def __init__(self, meta_learner):

        self.meta_learner = meta_learner
        self.meta_train_window = []
        self.meta_sel_window = []

        self.meta_data = []

    def generate_meta_data_train(X, y):
        # TODO: write function for determining if there is a potential outlier or not
        avg_X = np.mean(X)
        var_X = np.var(X)
        minimum_X = np.min(X)
        maximum_X = np.max(X)
        median_X = np.median(X)

        avg_y = np.mean(X)
        var_y = np.var(X)
        minimum_y = np.min(X)
        maximum_y = np.max(X)
        median_y = np.median(X)

        corr_coef_X_y = np.corrcoef(X, y) # correlation coefficient of attributes with

    def generate_meta_data_sel(X):
        # TODO: write function for determining if there is a potential outlier or not
        avg_X = np.mean(X)
        var_X = np.var(X)
        minimum_X = np.min(X)
        maximum_X = np.max(X)
        median_X = np.median(X)

        corr_coef_X = np.corrcoef(X) # correlation coefficient of attributes

        skew_X = scipy.stats.skew(X)
        kurtosis(X)


