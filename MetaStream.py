
class MetaStream():
    def __init__(self, metaLearner, learners, train_window_size, sel_window_size):
        
        # meta-learner
        self.metaLearner = metaLearner

        # regression models
        self.learners = learners

        self.train_window_size = train_window_size
        self.sel_window_size = sel_window_size


    def base_fit(self):
        """
        base-learning needs to be run before the initiation of the meta-learner.
        
        generates:
            - generate the base data for the meta-learner
        
        parameters:
        ----------
            X : 

            y : 

        returns:
        ----------
            None
        """


    def base_predict(self, X):
        """
        predict class value for X.

        parameters:
        ----------
            X : 

        returns:
        ----------
            y : 
        """

    # def initial_fit(self, X, y):
    #     """
    #     parameters:
    #     ----------
    #         X : 

    #         y : 

    #     returns:
    #     ----------
    #         None
    #     """


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
