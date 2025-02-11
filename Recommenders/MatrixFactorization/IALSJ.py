import implicit
from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
import numpy as np
from Recommenders.Recommender_utils import check_matrix


def linear_scaling_confidence(URM_train, alpha, epsilon):
    C = check_matrix(URM_train, format="csr", dtype=np.float32)
    C.data = 1.0 + alpha * C.data
    return C


def log_scaling_confidence(URM_train, alpha, epsilon):
    C = check_matrix(URM_train, format="csr", dtype=np.float32)
    C.data = 1.0 + alpha * np.log(1.0 + C.data / epsilon)
    return C


class ImplicitALSRecommender(BaseMatrixFactorizationRecommender):
    """ImplicitALSRecommender recommender"""

    RECOMMENDER_NAME = "ImplicitALSRecommender"

    def fit(self,
            num_factors=100,
            reg=0.01,
            use_native=True, use_cg=True, use_gpu=True,
            epochs=15,
            calculate_training_loss=False, num_threads=0, alpha=1.0, epsilon=1.0,
            confidence_scaling="linear", evaluator=None,
            patience=5, min_delta=1e-4, verbose=True):
        """
        Fit the model with early stopping based on validation MAP.

        :param evaluator: Evaluator instance for validation.
        :param validation_URM: Validation matrix for early stopping.
        :param patience: Number of epochs to wait without improvement.
        :param min_delta: Minimum improvement in MAP to reset patience.
        :param verbose: Print progress messages.
        """
        print (evaluator is None) 
        if evaluator is None :
            raise ValueError("Evaluator and validation_URM must be provided for early stopping.")

        print(f"Using GPU: {use_gpu}")

        conf_scale = linear_scaling_confidence if confidence_scaling == "linear" else log_scaling_confidence

        self.rec = implicit.als.AlternatingLeastSquares(factors=num_factors, regularization=reg,
                                                        use_native=use_native, use_cg=use_cg, use_gpu=use_gpu,
                                                        iterations=1,  # Itera manualmente
                                                        calculate_training_loss=calculate_training_loss,
                                                        num_threads=num_threads)

        best_map = -np.inf
        no_improvement_epochs = 0

        for epoch in range(epochs):
            # Fit one epoch
            self.rec.fit(conf_scale(self.URM_train, alpha, epsilon), show_progress=verbose)

            # Move to CPU if using GPU
            if use_gpu:
                self.rec = self.rec.to_cpu()

            self.USER_factors = self.rec.user_factors
            self.ITEM_factors = self.rec.item_factors

            # Evaluate MAP on validation set
            results, _ = evaluator.evaluateRecommender(self)
            current_map = results.loc[10]["MAP"]

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Validation MAP@10: {current_map:.6f}")

            # Early stopping logic
            if current_map > best_map + min_delta:
                best_map = current_map
                no_improvement_epochs = 0
                if verbose:
                    print("New best MAP! Saving the model.")
                # Save the best factors
                self.best_USER_factors = self.USER_factors.copy()
                self.best_ITEM_factors = self.ITEM_factors.copy()
            else:
                no_improvement_epochs += 1
                if verbose:
                    print(f"No improvement. Patience: {no_improvement_epochs}/{patience}")

            if no_improvement_epochs >= patience:
                if verbose:
                    print("Early stopping triggered.")
                break

        # Load the best model
        self.USER_factors = self.best_USER_factors
        self.ITEM_factors = self.best_ITEM_factors
