import numpy as np
from typing import List, Union

class QuantumKernelRegression2:
    """
    Implements a quantum kernel regression model.
    Supports SWAP test/Loschmidt Echo test and finite measurement statistics (shot noise).
    """
    def __init__(self, regularization_lambda: float = 1e-6, num_shots: int = None):
        self.regularization_lambda = regularization_lambda
        self.num_shots = num_shots
        self.train_density_matrices = None
        self.train_labels = np.array([])
        self.K_inv = np.array([]) 
        self.alpha = np.array([]) 
        self.kernel_type = "trace"
        self.kernel_matrix = None 

    def _apply_noise_vec(self, probs: np.ndarray) -> np.ndarray:
        """Applies vectorized binomial shot noise to a full matrix of probabilities simultaneously."""
        probs = np.clip(probs, 0.0, 1.0)
        
        if self.num_shots is None or self.num_shots == 0:
            return probs
            
        if self.kernel_type == "swap":
            successes = np.random.binomial(n=self.num_shots, p=0.5 * (1.0 + probs))
            return (2 * successes / self.num_shots) - 1
        elif self.kernel_type in ["trace", "le", "loschmidt"]:
            successes = np.random.binomial(n=self.num_shots, p=probs)
            return successes / self.num_shots
        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")

    def fit_vec(self, density_matrices_training: Union[List[np.ndarray], np.ndarray], 
                  labels_training: Union[List[float], np.ndarray], 
                  kernel_type: str = "trace"):
        
        if density_matrices_training is None or len(density_matrices_training) == 0:
            raise ValueError("Training density matrices cannot be empty.")
            
        self.train_density_matrices = np.asarray(density_matrices_training)
        self.train_labels = np.asarray(labels_training)
        self.kernel_type = kernel_type.lower() 

        # --- MASSIVE SPEEDUP HERE ---
        # Instead of a nested double loop, we compute the Hilbert-Schmidt inner 
        # products for all pairs simultaneously using Einstein Summation
        # Tr(A @ B) for all pairs = np.einsum('iab,jba->ij', A, B)
        exact_Tr = np.real(np.einsum('iab,jba->ij', 
                                     self.train_density_matrices, 
                                     self.train_density_matrices))

        # Apply physics + noise to the entire matrix instantly
        gram_matrix = self._apply_noise_vec(exact_Tr)

        # Enforce exact symmetry (noise can break perfect symmetry slightly)
        # We simulate measuring pairs only once by taking the upper triangle
        i_upper = np.triu_indices(gram_matrix.shape[0])
        gram_matrix = gram_matrix + gram_matrix.T
        gram_matrix[np.diag_indices_from(gram_matrix)] /= 2
        
        # Regularization
        gram_matrix_reg = gram_matrix + self.regularization_lambda * np.eye(gram_matrix.shape[0])
        self.kernel_matrix = gram_matrix_reg  

        try:
            self.K_inv = np.linalg.pinv(gram_matrix_reg)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Gram matrix could not be inverted. Error: {e}")

        self.alpha = self.K_inv @ self.train_labels

    def get_kernel_matrix(self):
        if self.kernel_matrix is None:
            raise RuntimeError("Kernel matrix has not been computed yet.")
        return self.kernel_matrix

    def predict_vec(self, test_density_matrices: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        if self.K_inv.size == 0 or self.alpha.size == 0:
            raise RuntimeError("Model has not been fitted yet.")
        if test_density_matrices is None or len(test_density_matrices) == 0:
            return np.array([])

        test_dm = np.asarray(test_density_matrices)
        
        # Vectorized evaluation for test dataset
        exact_Tr = np.real(np.einsum('mab,nba->mn', test_dm, self.train_density_matrices))
        kernel_test_train = self._apply_noise_vec(exact_Tr)

        return kernel_test_train @ self.alpha