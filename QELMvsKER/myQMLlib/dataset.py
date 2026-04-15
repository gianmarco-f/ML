import numpy as np

#definition on Pauli matrices
I = np.array([[1, 0],
              [0, 1]], dtype=complex)

X = np.array([[0, 1],
              [1, 0]], dtype=complex)

Y = np.array([[0, -1j],
              [1j,  0]], dtype=complex)

Z = np.array([[1,  0],
              [0, -1]], dtype=complex)

#generate a random density matrix dim x dim
def random_density_matrix(dim):
    #generate a random complex matrix
    A = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
    #make it hermitian
    A = (A + A.conj().T) / 2
    #make it positive semidefinite
    A = A @ A.conj().T
    #normalize to have trace 1
    rho = A / np.trace(A)
    return rho



#function to generate a dataset of density matrices of a qubit and their corresponding labels, namely the expectation value of a pauli matrix
def generate_dataset_pauli(N_training, N_test, pauli_matrix):
    """
    Generatea dataset of random density matrices of a qubit and their corresponding expectation values of a given Pauli matrix.

    Args:
        N_training (int): Number of training samples to generate.
        N_test (int): Number of test samples to generate.
        pauli_matrix (numpy.ndarray): The Pauli matrix for which to compute the expectation values.

    Returns:
        rho_training (numpy.ndarray): An array of shape (N_training, 2, 2) containing the training density matrices.
        rho_test (numpy.ndarray): An array of shape (N_test, 2, 2) containing the test density matrices.

        expectation_training (numpy.ndarray): An array of shape (N_training,) containing the expectation values of the Pauli matrix for the training samples.
        expectation_test (numpy.ndarray): An array of shape (N_test,) containing the expectation values of the Pauli matrix for the test samples.

    """
    rho_training = np.array([random_density_matrix(2) for i in range(N_training)])
    rho_test = np.array([random_density_matrix(2) for i in range(N_test)])

    expectation_training = np.array([np.trace(rho @ pauli_matrix).real for rho in rho_training])
    expectation_test = np.array([np.trace(rho @ pauli_matrix).real for rho in rho_test])

    return rho_training, rho_test, expectation_training, expectation_test

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

        print("Density matrices generated.")

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

        print("Expectation values computed.")

    def save_dataset(self, filename):
        """
        Save the generated dataset to a file.
        :param filename: The name of the file to save the dataset to.
        """
        if self.rho_training is None or self.rho_test is None or self.expectation_training is None or self.expectation_test is None:
            raise ValueError("Dataset not fully generated yet. Call generate_density_matrices() and compute_expectation_values() first.")
        
        np.savez(filename, rho_training=self.rho_training, rho_test=self.rho_test, expectation_training=self.expectation_training, expectation_test=self.expectation_test)

        print(f"Dataset saved to {filename}.")

