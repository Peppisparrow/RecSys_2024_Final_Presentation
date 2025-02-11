from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Recommender_utils import similarityMatrixTopK, check_matrix
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
from sklearn.preprocessing import normalize
import numpy as np
import time
import scipy.sparse as sps



class EASE_R_Recommender(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "EASE_R_Recommender"

    def __init__(self, URM_train, sparse_threshold_quota=None, verbose=True):
        super(EASE_R_Recommender, self).__init__(URM_train, verbose=verbose)
        self.sparse_threshold_quota = sparse_threshold_quota
        self.B = None  # Inizializziamo B come None

    def fit(self, topK=None, l2_norm=1e3, normalize_matrix=False, evaluator=None, iterations=200, step=1):
        start_time = time.time()
        self._print("Fitting model... ")

        if normalize_matrix:
            # Normalize rows and then columns
            self.URM_train = normalize(self.URM_train, norm='l2', axis=1)
            self.URM_train = normalize(self.URM_train, norm='l2', axis=0)
            self.URM_train = sps.csr_matrix(self.URM_train)

        # Graham matrix is X^T X, compute dot product
        grahm_matrix = self.URM_train.T.dot(self.URM_train).toarray()

        diag_indices = np.diag_indices(grahm_matrix.shape[0])
        grahm_matrix[diag_indices] += l2_norm

        P = np.linalg.inv(grahm_matrix)

        # Compute B
        B = P / (-np.diag(P))
        B[diag_indices] = 0.0

        # Salviamo B come attributo della classe
        self.B = B

        new_time_value, new_time_unit = seconds_to_biggest_unit(time.time() - start_time)
        self._print("Fitting model... done in {:.2f} {}".format(new_time_value, new_time_unit))

        # Iterative sparsification and evaluation
        for i in range(iterations):
            if topK is not None:
                print(f"Computing topK={topK - i*step}")
                BK = similarityMatrixTopK(self.B, k=topK - i*step, use_absolute_values=True, verbose=False)
            else:
                BK = self.B

            if self._is_content_sparse_check(BK):
                self._print("Detected model matrix to be sparse, changing format.")
                self.W_sparse = check_matrix(BK, format='csr', dtype=np.float32)
            else:
                self.W_sparse = check_matrix(BK, format='npy', dtype=np.float32)
                self._W_sparse_format_checked = True
                self._compute_item_score = self._compute_score_W_dense

            # Evaluate the model
            result_df, _ = evaluator.evaluateRecommender(self)
            print(result_df.loc[10]["MAP"])

    def tuningK(self, topK, evaluator, iterations=200, step=50):
        if self.B is None:
            raise ValueError("The B matrix is not initialized. Call 'fit' before using 'tuningK'.")

        for i in range(iterations):
            if topK is not None:
                print(f"Computing topK={topK - i * step}")
                BK = similarityMatrixTopK(self.B, k=topK - i * step, use_absolute_values=True, verbose=False)
            else:
                BK = self.B

            if self._is_content_sparse_check(BK):
                self._print("Detected model matrix to be sparse, changing format.")
                self.W_sparse = check_matrix(BK, format='csr', dtype=np.float32)
            else:
                self.W_sparse = check_matrix(BK, format='npy', dtype=np.float32)
                self._W_sparse_format_checked = True
                self._compute_item_score = self._compute_score_W_dense

            # Evaluate the model
            result_df, _ = evaluator.evaluateRecommender(self)
            print(result_df.loc[10]["MAP"])

    def load_model(self, folder_path, file_name=None):
        super(EASE_R_Recommender, self).load_model(folder_path, file_name=file_name)

        # Check for and load the B matrix if saved
        try:
            self.B = np.load(f"{folder_path}/{file_name}_B.npy")
        except FileNotFoundError:
            self._print("B matrix not found during loading. Ensure it was saved correctly.")

        if not sps.issparse(self.W_sparse):
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense

    def save_model(self, folder_path, file_name=None):
        super(EASE_R_Recommender, self).save_model(folder_path, file_name=file_name)

        # Save the B matrix
        if self.B is not None:
            np.save(f"{folder_path}/{file_name}_B.npy", self.B)
    def _is_content_sparse_check(self, matrix):

            if self.sparse_threshold_quota is None:
                return False

            if sps.issparse(matrix):
                nonzero = matrix.nnz
            else:
                nonzero = np.count_nonzero(matrix)

            return nonzero / (matrix.shape[0]**2) <= self.sparse_threshold_quota



    def _compute_score_W_dense(self, user_id_array, items_to_compute = None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_profile_array = self.URM_train[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse)#.toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse)#.toarray()

        return item_scores




