from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor

import argparse
import pathlib


class MetaStream():
    """
    meta-learning recommendation algorithm for concept shift or drift in data streams.
    """


    def __init__(self, meta_learner, learners, train_window_size, sel_window_size):
        
        # meta-learner
        self.meta_learner = meta_learner

        # regression models
        self.learners = learners

        self.train_window_size = train_window_size
        self.sel_window_size = sel_window_size


        print("meta-learner: ", self.meta_learner)
        print("learner: ", self.learners)
        print("training window size: ", self.train_window_size)
        print("selection window size: ", self.sel_window_size)



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

        for learner in self.learner:
            learner.fit(X, y)


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', type=pathlib.Path)
    parser.add_argument('-training_window_size', default=100, type=int, nargs=1)
    parser.add_argument('-selection_window_size', default=10, type=int, nargs=1)

    train_window_size = parser.parse_args().training_window_size
    sel_window_size = parser.parse_args().selection_window_size

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

    # NOTE: creates meta object
    metas = MetaStream(meta_learner, models, train_window_size, sel_window_size)
