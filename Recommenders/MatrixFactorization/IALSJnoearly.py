import implicit
from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
import numpy as np
from Recommenders.Recommender_utils import check_matrix


def linear_scaling_confidence(URM_train, alpha,epsilon):

    C = check_matrix(URM_train, format="csr", dtype = np.float32)
    C.data = 1.0 + alpha*C.data

    return C

def log_scaling_confidence(URM_train, alpha,epsilon):

    C = check_matrix(URM_train, format="csr", dtype = np.float32)
    C.data = 1.0 + alpha * np.log(1.0 + C.data / epsilon)

    return C

class ImplicitALSRecommender(BaseMatrixFactorizationRecommender):
    """ImplicitALSRecommender recommender"""

    RECOMMENDER_NAME = "ImplicitALSRecommender"

    def _build_confidence_matrix(self, confidence_scaling):

        if confidence_scaling == 'linear':
            self.C = self._linear_scaling_confidence()
        else:
            self.C = self._log_scaling_confidence()

        self.C_csc= check_matrix(self.C.copy(), format="csc", dtype = np.float32)






    def fit(self,
            num_factors=100,
            reg=0.01,
            use_native=True, use_cg=True, use_gpu=True,
            epochs=15,
            calculate_training_loss=False, num_threads=0, alpha=1.0, epsilon = 1.0, confidence_scaling="linear"
            ):

        print(f"Using gpu: {use_gpu}")

        conf_scale = linear_scaling_confidence if confidence_scaling=="linear" else log_scaling_confidence

        self.rec = implicit.als.AlternatingLeastSquares(factors=num_factors, regularization=reg,
                                                        use_native=use_native, use_cg=use_cg, use_gpu=use_gpu,
                                                        iterations=epochs,
                                                        calculate_training_loss=calculate_training_loss,
                                                        num_threads=num_threads)
        self.rec.fit(conf_scale(self.URM_train, alpha, epsilon), show_progress=self.verbose)

        if(use_gpu):
            self.rec = self.rec.to_cpu()

        self.USER_factors = self.rec.user_factors
        self.ITEM_factors = self.rec.item_factors