import numpy as np
from .basics import I, X, Y, Z, random_density_matrix, generate_dataset_pauli

class QuantumDatasetGenerator:
    def __init__(self, N_training, N_test, pauli_matrix):
        """
        Initialize the dataset generator. At the moment thid class admits only single qubit datasets, so the pauli_matrix must be 2x2.
        :param N_training: Number of training samples to generate.
        :param N_test: Number of test samples to generate.
        :param pauli_matrix: The Pauli matrix for which to compute the expectation values.
        """
        self.N_training = N_training
        self.N_test = N_test
        self.pauli_matrix = pauli_matrix

        self.rho_training = None
        self.rho_test = None
        self.expectation_training = None
        self.expectation_test = None

    def generate_density_matrices(self):
        """
        Generate random density matrices for training and test sets.
        """
        self.rho_training = np.array([random_density_matrix(2) for i in range(self.N_training)])
        self.rho_test = np.array([random_density_matrix(2) for i in range(self.N_test)])


    def compute_expectation_values(self):
        """
        Compute the expectation values of Tr(rho @ Observable) for the training and test sets.
        """
        if self.rho_training is None or self.rho_test is None:
            raise ValueError("Density matrices not generated yet. Call generate_density_matrices() first.")
        if self.pauli_matrix.shape != (2,2):
            raise ValueError("Pauli matrix must be 2x2. At least until now only single qubit datasets are supported.")
        
        self.expectation_training = np.array([np.trace(rho @ self.pauli_matrix) for rho in self.rho_training])
        self.expectation_test = np.array([np.trace(rho @ self.pauli_matrix) for rho in self.rho_test])


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
        
        np.savez(filename, rho_training=self.rho_training, rho_test=self.rho_test, expectation_training=self.expectation_training, expectation_test=self.expectation_test)

        print(f"Dataset saved to {filename}.")

