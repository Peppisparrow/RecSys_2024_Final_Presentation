"""
Created on 23/03/2019

@author: Maurizio Ferrari Dacrema
"""



from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.Recommender_utils import check_matrix
import numpy as np


class IALSRecommender(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):
    """

    Binary/Implicit Alternating Least Squares (IALS)
    See:
    Y. Hu, Y. Koren and C. Volinsky, Collaborative filtering for implicit feedback datasets, ICDM 2008.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.5120&rep=rep1&type=pdf

    R. Pan et al., One-class collaborative filtering, ICDM 2008.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.4684&rep=rep1&type=pdf

    Factorization model for binary feedback.
    First, splits the feedback matrix R as the element-wise a Preference matrix P and a Confidence matrix C.
    Then computes the decomposition of them into the dot product of two matrices X and Y of latent factors.
    X represent the user latent factors, Y the item latent factors.

    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin}\limits_{x*,y*}\frac{1}{2}\sum_{i,j}{c_{ij}(p_{ij}-x_i^T y_j) + \lambda(\sum_{i}{||x_i||^2} + \sum_{j}{||y_j||^2})}
    """

    RECOMMENDER_NAME = "IALSRecommender"

    AVAILABLE_CONFIDENCE_SCALING = ["linear", "log"]


    def fit(self, epochs=300,
        num_factors=20,
        confidence_scaling="linear",
        alpha=1.0,
        epsilon=1.0,
        reg=1e-3,
        init_mean=0.0,
        init_std=0.1,
        evaluator=None,
        validation_metric="MAP",
        stop_on_validation=True,
        validation_every_n=1,
        lower_validations_allowed=5,
        **earlystopping_kwargs):
        """
        :param evaluator: The EvaluatorHoldOut instance for validation
        :param validation_metric: Metric to use for validation (e.g., "MAP")
        :param stop_on_validation: Whether to enable early stopping
        :param validation_every_n: Frequency of validation
        :param lower_validations_allowed: Number of times the validation can decrease consecutively
        """

        # Asserzioni per garantire parametri corretti
        if evaluator is not None:
            assert validation_metric is not None, "Validation metric must be provided if evaluator is defined."
            assert validation_every_n > 0, "validation_every_n must be a positive integer."
            if stop_on_validation:
                assert lower_validations_allowed is not None, "lower_validations_allowed must be defined for early stopping."

        self.num_factors = num_factors
        self.alpha = alpha
        self.epsilon = epsilon
        self.reg = reg
        self.evaluator = evaluator

        self.USER_factors = self._init_factors(self.n_users, False)  # don't need values, will compute them
        self.ITEM_factors = self._init_factors(self.n_items)

        self._build_confidence_matrix(confidence_scaling)

        warm_user_mask = np.ediff1d(self.URM_train.indptr) > 0
        warm_item_mask = np.ediff1d(self.URM_train.tocsc().indptr) > 0

        self.warm_users = np.arange(0, self.n_users, dtype=np.int32)[warm_user_mask]
        self.warm_items = np.arange(0, self.n_items, dtype=np.int32)[warm_item_mask]

        self.regularization_diagonal = np.diag(self.reg * np.ones(self.num_factors))

        self._update_best_model()

        # Chiamata al metodo di training con early stopping
        self._train_with_early_stopping(epochs_max=epochs,
                                        epochs_min=10,
                                        validation_every_n=validation_every_n,
                                        stop_on_validation=stop_on_validation,
                                        validation_metric=validation_metric,
                                        lower_validations_allowed=lower_validations_allowed,
                                        evaluator_object=evaluator,
                                        algorithm_name=self.RECOMMENDER_NAME)

        # Alla fine del training, aggiorna i fattori migliori
        self.USER_factors = self.USER_factors_best
        self.ITEM_factors = self.ITEM_factors_best





    def _build_confidence_matrix(self, confidence_scaling):

        if confidence_scaling == 'linear':
            self.C = self._linear_scaling_confidence()
        else:
            self.C = self._log_scaling_confidence()

        self.C_csc= check_matrix(self.C.copy(), format="csc", dtype = np.float32)




    def _linear_scaling_confidence(self):

        C = check_matrix(self.URM_train, format="csr", dtype = np.float32)
        C.data = 1.0 + self.alpha*C.data

        return C

    def _log_scaling_confidence(self):

        C = check_matrix(self.URM_train, format="csr", dtype = np.float32)
        C.data = 1.0 + self.alpha * np.log(1.0 + C.data / self.epsilon)

        return C




    def _prepare_model_for_validation(self):
        pass


    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()


    def _run_epoch(self, num_epoch):

        # fit user factors
        # VV = n_factors x n_factors
        VV = self.ITEM_factors.T.dot(self.ITEM_factors)

        for user_id in self.warm_users:
            # get (positive i.e. non-zero scored) items for user

            start_pos = self.C.indptr[user_id]
            end_pos = self.C.indptr[user_id + 1]

            user_profile = self.C.indices[start_pos:end_pos]
            user_confidence = self.C.data[start_pos:end_pos]

            self.USER_factors[user_id, :] = self._update_row(user_profile, user_confidence, self.ITEM_factors, VV)

        # fit item factors
        # UU = n_factors x n_factors
        UU = self.USER_factors.T.dot(self.USER_factors)

        for item_id in self.warm_items:

            start_pos = self.C_csc.indptr[item_id]
            end_pos = self.C_csc.indptr[item_id + 1]

            item_profile = self.C_csc.indices[start_pos:end_pos]
            item_confidence = self.C_csc.data[start_pos:end_pos]

            self.ITEM_factors[item_id, :] = self._update_row(item_profile, item_confidence, self.USER_factors, UU)



    def _update_row(self, interaction_profile, interaction_confidence, Y, YtY):
        """
        Update latent factors for a single user or item.

        Y = |n_interactions|x|n_factors|
        YtY =   |n_factors|x|n_factors|
        """

        # Latent factors ony of item/users for which an interaction exists in the interaction profile
        Y_interactions = Y[interaction_profile, :]

        # Following the notation of the original paper we report the update rule for the Item factors (User factors are identical):
        # Y are the item factors |n_items|x|n_factors|
        # Cu is a diagonal matrix |n_interactions|x|n_interactions| with the user confidence for the observed items
        # p(u) is a boolean vectors indexing only observed items. Here it will disappear as we already extract only the observed latent factors
        #       however, it will have an impact in the dimensions of the matrix, since it transforms Cu from a diagonal matrix to a row vector of 1 row and |n_interactions| columns
        # (Yt*Cu*Y + reg*I)^-1 * Yt*Cu*profile
        # which can be decomposed as
        # (YtY + Yt*(Cu-I)*Y + reg*I)^-1 * Yt*Cu*p(u)

        # A = (|n_interactions|x|n_factors|) dot (|n_interactions|x|n_interactions| ) dot (|n_interactions|x|n_factors| )
        #   = |n_factors|x|n_factors|
        # A_slow = Y_interactions.T.dot(np.diag(interaction_confidence - 1)).dot(Y_interactions)

        # if v = diag(|n_interactions|) and k = |n_interactions|x|n_factors|
        # computing np.diag(v).dot(k) will be SLOW
        # we use an equivalent formulation (v * k.T).T which is much faster
        A = Y_interactions.T.dot(((interaction_confidence - 1) * Y_interactions.T).T)

        B = YtY + A + self.regularization_diagonal

        return np.dot(np.linalg.inv(B), Y_interactions.T.dot(interaction_confidence))


    def _init_factors(self, num_factors, assign_values=True):

        if assign_values:
            return self.num_factors**-0.5*np.random.random_sample((num_factors, self.num_factors))

        else:
            return np.empty((num_factors, self.num_factors))



