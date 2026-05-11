import numpy as np
from typing import List, Union, Tuple

class QuantumKernelRegression:
    """
    Implements a quantum kernel regression model.
    Supports SWAP test/Loschmidt Echo test and finite measurement statistics (shot noise).
    """
    def __init__(self, regularization_lambda: float = 1e-6, num_shots: int = None):
        """
        Args:
            regularization_lambda: Adds to the diagonal to ensure invertibility.
                                   If there is no regularization, the kernel matrix can become singular
                                   
            num_shots: Number of measurements for finite statistics. 
                       If None, computes exact theoretical expectations.
        """
        self.regularization_lambda = regularization_lambda
        self.num_shots = num_shots
        self.train_density_matrices = None
        self.train_labels = np.array([])
        self.K_inv = np.array([]) 
        self.alpha = np.array([]) 
        self.kernel_type = "trace"
        self.kernel_matrix = None 

    def _HS_inner_product(self, rho1, rho2):
        """
        Computes the Hilbert-Schmidt inner product between two density matrices rho1 and rho2.
        """
        #matrix multiplication and trace
        return np.trace(rho1 @ rho2)

    def _kernel_evaluation_with_noise(self, HS_rho1_rho2: float) -> float:
        """
        Applies the correct physical measurement formula and shot noise
        based on the chosen quantum circuit protocol.
        """
        # Exact Tr[rho1 rho2] should strictly be between 0 and 1
        # Clipping handles tiny floating-point errors (e.g., 1.000000002)
        HS_rho1_rho2 = np.clip(HS_rho1_rho2, 0.0, 1.0)
        

        # Apply Shot Noise if num_shots is specified
        if self.num_shots is None or self.num_shots == 0:
            # Theoretical limit (infinite shots):
            return HS_rho1_rho2
        else:
            # Finite shots (Binomial sampling simulating the circuit measurements)
            if self.kernel_type == "swap":
                # For SWAP test, we measure a single ancilla qubit, we add finite shot to the probability
                # of measuring 0, then we get back the noisy kernel value 
                measured_successes = np.random.binomial(n=self.num_shots, p = 0.5 * (1.0 + HS_rho1_rho2))
                return (2 * measured_successes / self.num_shots) - 1  # Rescale back to [-1, 1]
            
            elif self.kernel_type in ["trace", "le", "loschmidt"]:
                # For Loschmidt Echo, we measure all qubits and only the all-0 outcome corresponds to success
                measured_successes = np.random.binomial(n=self.num_shots, p=HS_rho1_rho2)
                return measured_successes / self.num_shots
            
            else:
                raise ValueError(f"Unknown kernel_type: {self.kernel_type}")    

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

    def fit(self, density_matrices_training: Union[List[np.ndarray], np.ndarray], 
                  labels_training: Union[List[float], np.ndarray], 
                  kernel_type: str = "trace"):
        
        if density_matrices_training is None or len(density_matrices_training) == 0:
            raise ValueError("Training density matrices cannot be empty.")
        if len(density_matrices_training) != len(labels_training):
            raise ValueError("Number of density matrices must match number of labels.")

        self.train_density_matrices = density_matrices_training
        self.train_labels = np.asarray(labels_training)
        self.kernel_type = kernel_type.lower() #fix the kernel type to be lowercase to avoid user errors

        n_samples = len(self.train_density_matrices)
        gram_matrix = np.zeros((n_samples, n_samples), dtype=float)

        for i in range(n_samples):
            # We measure each pair (i, j) only once to save shots and ensure the 
            # resulting estimated matrix is perfectly symmetric!
            for j in range(i, n_samples): 
                exact_Tr = self._HS_inner_product(self.train_density_matrices[i],
                                                          self.train_density_matrices[j])
            
                # Apply physics + noise
                val = self._kernel_evaluation_with_noise(exact_Tr)
                
                gram_matrix[i, j] = val
                if i != j:
                    gram_matrix[j, i] = val

        # Regularization is EXTRA important with shot noise
        gram_matrix_reg = gram_matrix + self.regularization_lambda * np.eye(n_samples)
        self.kernel_matrix = gram_matrix_reg  # Store the regularized kernel matrix for potential debugging/analysis

        try:
            # We use pseudo-inverse to handle potential numerical issues, but we also catch exceptions to provide clearer error messages
            self.K_inv = np.linalg.pinv(gram_matrix_reg)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Gram matrix could not be inverted. Error: {e}")

        self.alpha = self.K_inv @ self.train_labels

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
        gram_matrix = np.triu(gram_matrix)
        gram_matrix = gram_matrix + gram_matrix.T - np.diag(np.diag(gram_matrix))
        
        
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
            raise RuntimeError("Kernel matrix has not been computed yet. Call fit() first.")
        return self.kernel_matrix

    def predict(self, test_density_matrices: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        
        if self.K_inv.size == 0 or self.alpha.size == 0:
            raise RuntimeError("Model has not been fitted yet.")
        if test_density_matrices is None or len(test_density_matrices) == 0:
            return np.array([])

        n_test_samples = len(test_density_matrices)
        n_train_samples = len(self.train_density_matrices)
        
        kernel_test_train = np.zeros((n_test_samples, n_train_samples), dtype=float)

        for m in range(n_test_samples):
            for i in range(n_train_samples):
                exact_Tr = self._HS_inner_product(test_density_matrices[m],
                                                          self.train_density_matrices[i])
                
                kernel_test_train[m, i] = self._kernel_evaluation_with_noise(exact_Tr)

        return kernel_test_train @ self.alpha
    
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