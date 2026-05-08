import numpy as np
from typing import List, Tuple, Union

class QuantumExtremeLearningMachine:
    """
    Implements a Quantum Extreme Learning Machine (QELM) for regression tasks.
    The QELM to maps classical data into a high-dimensional feature space,
    and then appliesa measurement and a linear regression in the measurement outcomes.
    """
    def __init__(self,
                 isometry: np.ndarray,
                 povm: List[np.ndarray],
                 bipartite_dims: Tuple[int, int],
                 keep_subsystem: int = 1,
                 num_shots: int = None):
        """
        Args:
            isometry: The V matrix defining the unitary embedding.
            povm: A list of measurement operators for the reservoir.
            bipartite_dims: A tuple (dim0, dim1) representing the composite space
                            after the isometry (e.g., [2, 32] for d_out=64).
            keep_subsystem: Index (0 or 1) of the subsystem to KEEP after partial trace.
                            E.g., if dims=[2, 32] and keep=1, the 32-dim system is kept.
            num_shots: Number of measurements for finite statistics (None for exact).
        """
        self.isometry = isometry
        self.povm = povm
        self.povm_array = np.array(povm)  # Convert to array
        self.bipartite_dims = bipartite_dims
        self.keep_subsystem = keep_subsystem
        self.num_shots = num_shots

        self.W = np.array([]) # Placeholder for trained readout weights
        self.num_povm_elements = len(povm)

    def _partial_trace(self, rho):
        """
        Fast pure-Numpy implementation of partial trace using Einstein summation.
        
        Args:
            rho: The input density matrix (2D array).
        
        Returns the reduced density matrix after tracing out the other subsystem.
        """
        d0, d1 = self.bipartite_dims
        rho_tensor = rho.reshape(d0, d1, d0, d1)
        if self.keep_subsystem == 0:
        # Keep subsystem 0, trace out subsystem 1
            reduced_rho = np.einsum('ijkj->ik', rho_tensor)
        elif self.keep_subsystem == 1: 
        # Keep subsystem 1, trace out subsystem 0
            reduced_rho = np.einsum('ijil->jl', rho_tensor)
        else:
            raise ValueError("Invalid 'keep_subsystem' index. Must be 0 or 1.")
        return reduced_rho
    
    def _evolve_and_measure(self,rho):
        """
        Evolves the state through the reservoir and applies the POVM measurement.
        """

        # Evolve: V @ rho @ V^dagger
        rho_evolved = self.isometry @ rho @ self.isometry.conj().T

        # Partial trace to get the reduced state of the subsystem we keep
        rho_reduced = self._partial_trace(rho_evolved)

        # Compute theoretical probabilities for each POVM element
        probs = np.array([np.real(np.trace(M @ rho_reduced)) for M in self.povm])

        #Numerical stability: clip probabilities to [0,1] and renormalize
        probs = np.clip(probs, 0.0, 1.0)
        probs = probs /np.sum(probs)

        #Shot noise
        if self.num_shots is None or self.num_shots == 0:
            return probs
        else:
            counts = np.random.multinomial(self.num_shots, probs)
            return counts / self.num_shots
        

    def get_features(self, density_matrices: List[np.ndarray]) -> np.ndarray:
        """
        Passes a list of density matrices through the reservoir and returns 
        the probability matrix P (Features).
        Returns: matrix of shape (N_samples, N_povm_elements)
        """
        N_samples = len(density_matrices)
        P_matrix = np.zeros((N_samples, self.num_povm_elements))
        
        for idx, rho in enumerate(density_matrices):
            P_matrix[idx, :] = self._evolve_and_measure(rho)
            
        return P_matrix
    

    def get_features_vec(self, density_matrices: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Passes a batch of density matrices through the reservoir simultaneously.
        """
        rho = np.asarray(density_matrices)
        N_samples = rho.shape[0]
        
        # 1. EVOLVE (Batch mode)
        # V @ rho @ V^dagger implemented via tensor contraction
        rho_evolved = np.einsum('oa,nab,pb->nop', self.isometry, rho, self.isometry.conj())

        # 2. PARTIAL TRACE (Batch mode)
        d0, d1 = self.bipartite_dims
        rho_tensor = rho_evolved.reshape(N_samples, d0, d1, d0, d1)
        
        if self.keep_subsystem == 0:
            # Trace out subsystem 1
            rho_reduced = np.einsum('nijkj->nik', rho_tensor)
        elif self.keep_subsystem == 1: 
            # Trace out subsystem 0
            rho_reduced = np.einsum('nijil->njl', rho_tensor)
        else:
            raise ValueError("Invalid 'keep_subsystem' index.")

        # 3. MEASURE (Batch mode)
        # Tr(M @ rho_reduced) for all POVMs and all samples
        probs = np.real(np.einsum('mab,nba->nm', self.povm_array, rho_reduced))

        # Numerical stability: clip and renormalize along the POVM axis
        probs = np.clip(probs, 0.0, 1.0)
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        # 4. APPLY SHOT NOISE
        if self.num_shots is None or self.num_shots == 0:
            return probs
        else:
            # np.random.multinomial requires iterating over the sample axis 
            # (which is small enough to be extremely fast)
            counts = np.array([np.random.multinomial(self.num_shots, p) for p in probs])
            return counts / self.num_shots
    
    def fit(self, density_matrices: List[np.ndarray], labels: Union[List[float], np.ndarray]):
        """Trains the QELM linear readout using Moore-Penrose pseudo-inverse."""
        labels = np.asarray(labels)
        
        # Get probability vectors (Features matrix P)
        P_train = self.get_features(density_matrices)
        
        # Standard ML solver: W = pinv(P) @ y
        # Note: P is (N_train, N_povm), y is (N_train,) -> W is (N_povm,)
        self.W = np.linalg.pinv(P_train) @ labels

    def fit_vec(self, density_matrices: Union[List[np.ndarray], np.ndarray], labels: Union[List[float], np.ndarray]):
        labels = np.asarray(labels)
        P_train = self.get_features_vec(density_matrices)
        self.W = np.linalg.pinv(P_train) @ labels



    def predict(self, new_density_matrices: List[np.ndarray]) -> np.ndarray:
        """Predicts the labels for new density matrices."""
        if self.W.size == 0:
            raise RuntimeError("Model has not been fitted yet.")
            
        # Get probability vectors for test set
        P_test = self.get_features(new_density_matrices)
        
        # Predict: y_hat = P_test @ W
        return P_test @ self.W
    
    def predict_vec(self, new_density_matrices: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        if self.W.size == 0:
            raise RuntimeError("Model has not been fitted yet.")
        P_test = self.get_features_vec(new_density_matrices)
        return P_test @ self.W