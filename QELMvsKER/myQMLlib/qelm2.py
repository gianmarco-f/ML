import numpy as np
from typing import List, Tuple, Union

class QuantumExtremeLearningMachine:
    def __init__(self,
                 isometry: np.ndarray,
                 povm: List[np.ndarray],
                 bipartite_dims: Tuple[int, int],
                 keep_subsystem: int = 1,
                 num_shots: int = None):
        self.isometry = isometry
        self.povm = povm
        self.povm_array = np.array(povm)  # Convert to array for vectorization
        self.bipartite_dims = bipartite_dims
        self.keep_subsystem = keep_subsystem
        self.num_shots = num_shots

        self.W = np.array([]) 
        self.num_povm_elements = len(povm)

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
    
    def fit_vec(self, density_matrices: Union[List[np.ndarray], np.ndarray], labels: Union[List[float], np.ndarray]):
        labels = np.asarray(labels)
        P_train = self.get_features_vec(density_matrices)
        self.W = np.linalg.pinv(P_train) @ labels

    def predict_vec(self, new_density_matrices: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        if self.W.size == 0:
            raise RuntimeError("Model has not been fitted yet.")
        P_test = self.get_features_vec(new_density_matrices)
        return P_test @ self.W