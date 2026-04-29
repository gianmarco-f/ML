import numpy as np
# Assuming random_density_matrix handles dimension 'd' correctly:
from .basics import random_density_matrix 

class QuantumDatasetGenerator2:
    def __init__(self, N_training, N_test, observable):
        """
        Initialize the dataset generator for qudits of dimension d (= 2^n).
        :param N_training: Number of training samples to generate.
        :param N_test: Number of test samples to generate.
        :param observable: The d x d Hermitian matrix (observable) to compute expectation values for.
        """
        self.N_training = N_training
        self.N_test = N_test
        self.observable = np.array(observable)
        
        # Infer the dimension 'd' from the observable matrix
        self.d = self.observable.shape[0]
        
        # Sanity check: Ensure it's a square matrix
        if self.observable.shape[0] != self.observable.shape[1]:
            raise ValueError("Observable must be a square matrix.")
            
        self.rho_training = None
        self.rho_test = None
        self.expectation_training = None
        self.expectation_test = None

    def generate_density_matrices(self):
        """
        Generate random density matrices of dimension d for training and test sets.
        """
        # Pass the dynamic dimension 'self.d' to the random_density_matrix function
        self.rho_training = np.array([random_density_matrix(self.d) for _ in range(self.N_training)])
        self.rho_test = np.array([random_density_matrix(self.d) for _ in range(self.N_test)])

    def compute_expectation_values(self):
        """
        Compute the expectation values of Tr(rho @ Observable) for the training and test sets.
        """
        if self.rho_training is None or self.rho_test is None:
            raise ValueError("Density matrices not generated yet. Call generate_density_matrices() first.")
        
        # Compute Tr(rho @ observable). 
        # We use np.real() because floating point math often leaves microscopic complex residuals (e.g. + 0.00000001j).
        # QML models require strictly float/real inputs.
        self.expectation_training = np.real(
            np.array([np.trace(rho @ self.observable) for rho in self.rho_training])
        )
        self.expectation_test = np.real(
            np.array([np.trace(rho @ self.observable) for rho in self.rho_test])
        )

    def get_training_dataset(self):
        return self.rho_training, self.expectation_training
    
    def get_test_dataset(self):
        return self.rho_test, self.expectation_test

    def save_dataset(self, filename):
        """
        Save the generated dataset to a file.
        :param filename: The name of the file to save the dataset to.
        """
        if self.rho_training is None or self.rho_test is None or self.expectation_training is None or self.expectation_test is None:
            raise ValueError("Dataset not fully generated yet. Call generate_density_matrices() and compute_expectation_values() first.")
        
        np.savez(filename, 
                 rho_training=self.rho_training, 
                 rho_test=self.rho_test, 
                 expectation_training=self.expectation_training, 
                 expectation_test=self.expectation_test)

        print(f"Dataset saved to {filename}.")