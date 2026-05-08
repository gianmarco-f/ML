import numpy as np
from .baisics2 import random_density_matrices

class QuantumDatasetGenerator:
    def __init__(self, N_training, N_test, pauli_matrix):
        self.N_training = N_training
        self.N_test = N_test
        self.pauli_matrix = pauli_matrix

        self.rho_training = None
        self.rho_test = None
        self.expectation_training = None
        self.expectation_test = None

    def generate_density_matrices(self):
        # Uses the new vectorized backend in C
        self.rho_training = random_density_matrices(2, self.N_training)
        self.rho_test = random_density_matrices(2, self.N_test)

    def compute_expectation_values(self):
        if self.rho_training is None or self.rho_test is None:
            raise ValueError("Density matrices not generated yet.")
        
        # Highly optimized vectorized trace operation 
        self.expectation_training = np.real(np.einsum('nij,ji->n', self.rho_training, self.pauli_matrix))
        self.expectation_test = np.real(np.einsum('nij,ji->n', self.rho_test, self.pauli_matrix))

    def get_training_dataset(self):
        return self.rho_training, self.expectation_training
    
    def get_test_dataset(self):
        return self.rho_test, self.expectation_test

    def save_dataset(self, filename):
        if self.rho_training is None or self.rho_test is None:
            raise ValueError("Dataset not fully generated yet.")
        np.savez(filename, rho_training=self.rho_training, rho_test=self.rho_test, 
                 expectation_training=self.expectation_training, expectation_test=self.expectation_test)
        print(f"Dataset saved to {filename}.")