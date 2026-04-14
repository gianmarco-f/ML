
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Tuple

#definition on Pauli matrices
I = np.array([[1, 0],
              [0, 1]], dtype=complex)

X = np.array([[0, 1],
              [1, 0]], dtype=complex)

Y = np.array([[0, -1j],
              [1j,  0]], dtype=complex)

Z = np.array([[1,  0],
              [0, -1]], dtype=complex)



def  generate_dataset_pauli(N_training, N_test):
    """
    Generate a data of random quantum states (qubits) and their corresponding expectation values of the Pauli matrices.

    Args:
        N_training (int): Number of training samples to generate.
        N_test (int): Number of test samples to generate.

    Returns:
        rho_training (numpy.array): Array of shape (N_training, 2, 2) containing the density matrices of the training samples.
        rho_test (numpy.array): Array of shape (N_test, 2, 2) containing the density matrices of the test samples.

        expe_training_X (numpy.array): Array of shape (N_training) containing the expectation values of the X Pauli matrix for the  training samples.
        expe_training_Y (numpy.array): Array of shape (N_training) containing the expectation values of the Y Pauli matrix for the  training samples.
        expe_training_Z (numpy.array): Array of shape (N_training) containing the expectation values of the Z Pauli matrix for the  training samples.
        expe_test_X (numpy.array): Array of shape (N_test) containing the expectation values of the X Pauli matrix for the  test samples.
        expe_test_Y (numpy.array): Array of shape (N_test) containing the expectation values of the Y Pauli matrix for the  test samples.
        expe_test_Z (numpy.array): Array of shape (N_test) containing the expectation values of the Z Pauli matrix for the  test samples.

    """

    #density matrices for training and test sets
    rho_training = np.zeros((N_training, 2, 2), dtype=complex)
    rho_test = np.zeros((N_test,2,2), dtype=complex)

    #expectationvalues for training and test sets
    expe_training_X = np.zeros(N_training)
    expe_test_X = np.zeros(N_test)

    expe_training_Y = np.zeros(N_training)
    expe_test_Y = np.zeros(N_test)

    expe_training_Z = np.zeros(N_training)
    expe_test_Z = np.zeros(N_test)

    #generate random density matrices and compute expectation values
    for i in range(N_training):
        #sample a random Bloch vector uniformly in the Bloch ball
        r = np.random.normal(size=3)
        r = r / np.linalg.norm(r) * np.random.rand()**(1/3)
        
        #construct density matrix
        rho_training[i] = 0.5 * (I + r[0]*X + r[1]*Y + r[2]*Z)

        #expectation value of Pauli X
        expe_training_X[i] = np.real(np.trace(rho_training[i] @ X))
        expe_training_Y[i] = np.real(np.trace(rho_training[i] @ Y))
        expe_training_Z[i] = np.real(np.trace(rho_training[i] @ Z))

    for i in range(N_test):
        #sample a random Bloch vector uniformly in the Bloch ball
        r = np.random.normal(size=3)
        r = r / np.linalg.norm(r) * np.random.rand()**(1/3)
        
        #construct density matrix
        rho_test[i] = 0.5 * (I + r[0]*X + r[1]*Y + r[2]*Z)

        #cxpectation value of Pauli X
        expe_test_X[i] = np.real(np.trace(rho_test[i] @ X))
        expe_test_Y[i] = np.real(np.trace(rho_test[i] @ Y))
        expe_test_Z[i] = np.real(np.trace(rho_test[i] @ Z))

    return rho_training, rho_test, expe_training_X, expe_training_Y, expe_training_Z, expe_test_X, expe_test_Y, expe_test_Z



def generate_random_density_matrix(dim):
    # Create a random complex matrix G
    G = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim))
    
    # Compute G * G_dagger (makes it Hermitian and Positive)
    A = G @ G.conj().T
    
    # Normalize the trace to 1
    rho = A / np.trace(A)
    return rho




def density_to_bloch(rho):
    x = np.real(np.trace(rho @ X))
    y = np.real(np.trace(rho @ Y))
    z = np.real(np.trace(rho @ Z))
    return np.array([x, y, z])



def plot_bloch_sphere(points):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    # Draw sphere surface
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.1)

    # Draw axes
    ax.quiver(0,0,0, 1,0,0, color='r')
    ax.quiver(0,0,0, 0,1,0, color='g')
    ax.quiver(0,0,0, 0,0,1, color='b')

    # Plot points
    points = np.array(points)
    ax.scatter(points[:,0], points[:,1], points[:,2], 
               color='black', s=40)

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_box_aspect([1,1,1])
    plt.show()



def quantum_kernel(rho1: np.ndarray, rho2: np.ndarray) -> complex:
    """
    Computes the quantum kernel K(rho1, rho2) = Tr[rho1 @ rho2].
    Assumes rho1 and rho2 are valid density matrices.
    """
    # Matrix multiplication followed by trace
    return np.trace(rho1 @ rho2)



class QuantumKernelRegression:
    """
    Implements a kernel regression model using a quantum kernel.
    f(rho) = k(rho_i, rho) @ K_inv @ y_j
    """
    def __init__(self, regularization_lambda: float = 1e-6):
        """
        Initializes the QuantumKernelRegression model.

        Args:
            regularization_lambda (float): Regularization parameter (lambda)
                                           for Tikhonov regularization.
                                           Added to the diagonal of the Gram matrix
                                           to ensure invertibility and stability.
        """
        self.regularization_lambda = regularization_lambda
        self.train_density_matrices: List[np.ndarray] = []
        self.train_labels: np.ndarray = np.array([])
        self.K_inv: np.ndarray = np.array([]) # Inverse of the (regularized, actually maybe I won't regularize) Gram matrix
        self.alpha: np.ndarray = np.array([]) # Dual variables (K_inv @ y)

    def fit(self, density_matrices: List[np.ndarray], labels: Union[List[float], np.ndarray]):
        """
        Trains the quantum kernel model by computing the Gram matrix,
        inverting it, and calculating the dual variables (alpha).

        Args:
            density_matrices (List[np.ndarray]): A list of training density matrices.
                                                 Each element is a 2D NumPy array.
            labels (Union[List[float], np.ndarray]): Corresponding labels (y_j) for
                                                    the training density matrices.
        """
        if density_matrices is None or len(density_matrices) == 0:
            raise ValueError("Training density matrices cannot be empty.")
        if len(density_matrices) != len(labels):
            raise ValueError("Number of density matrices must match number of labels.")

        self.train_density_matrices = density_matrices
        self.train_labels = np.asarray(labels) # Ensure labels are a numpy array

        n_samples = len(self.train_density_matrices)
        
        # Initialize Gram matrix. Density matrices can be complex, so K should be complex.
        gram_matrix = np.zeros((n_samples, n_samples), dtype=complex)

        print(f"Computing Gram matrix of size {n_samples}x{n_samples}...")
        # Compute the Gram matrix K_ij = Tr[rho_i @ rho_j]
        for i in range(n_samples):
            for j in range(n_samples):
                gram_matrix[i, j] = quantum_kernel(self.train_density_matrices[i],
                                                   self.train_density_matrices[j])
                
        #In this section there is an optimization for the computation of the Gram matrix, due to the symmetry of the kernel
        #for i in range(n_samples):
            #for j in range(i, n_samples): # Notice j starts from i
                # Compute kernel and take .real to cast away the 0j imaginary part safely
                #val = np.real(quantum_kernel(self.train_density_matrices[i], self.train_density_matrices[j]))
                
                #gram_matrix[i, j] = val
                
                # Mirror the value to the lower triangle if it's not the diagonal
                #if i != j:
                    #gram_matrix[j, i] = val

        print("Gram matrix computation complete.")

        # Apply Tikhonov regularization: K_reg = K + lambda * I
        gram_matrix_reg = gram_matrix + self.regularization_lambda * np.eye(n_samples)

        print("Inverting Gram matrix...")
        # Invert the regularized Gram matrix
        try:
            self.K_inv = np.linalg.inv(gram_matrix_reg)
            print("Gram matrix inverted successfully.")
        except np.linalg.LinAlgError as e:
            print(f"Error: Gram matrix could not be inverted. It might be singular or ill-conditioned even with regularization. Error: {e}")
            raise

        # Calculate the dual variables alpha = K_inv @ y
        # This is equivalent to solving K @ alpha = y
        self.alpha = self.K_inv @ self.train_labels
        print("Model fitted: alpha (dual variables) computed.")

    def predict(self, new_density_matrices: List[np.ndarray]) -> np.ndarray:
        """
        Predicts the output value f(rho) for new density matrices.

        f(rho) = k(rho_i, rho) @ alpha

        Args:
            new_density_matrices (List[np.ndarray]): A list of new density matrices
                                                    for which to make predictions.

        Returns:
            np.ndarray: An array of predicted values for each new density matrix.
        """
        if not self.K_inv.size or not self.alpha.size:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        if new_density_matrices is None or len(new_density_matrices) == 0:
            return np.array([])

        n_test_samples = len(new_density_matrices)
        n_train_samples = len(self.train_density_matrices)
        
        # Calculate the kernel matrix between training and new test samples
        # K_test_train[m, i] = k(rho_test_m, rho_train_i)
        kernel_test_train = np.zeros((n_test_samples, n_train_samples), dtype=complex)

        print(f"Computing kernel matrix for {n_test_samples} test samples against {n_train_samples} training samples...")
        for m in range(n_test_samples):
            for i in range(n_train_samples):
                kernel_test_train[m, i] = quantum_kernel(new_density_matrices[m],
                                                         self.train_density_matrices[i])
        print("Kernel matrix computation for test samples complete.")

        # Predict f(rho) = K_test_train @ alpha
        # The result might be complex if K_inv or y are complex, but typically y is real.
        predictions = kernel_test_train @ self.alpha


        print("Prediction dimension is")
        print(predictions.shape)
        # If labels are real, predictions should ideally be real.
        # Taking the real part is often appropriate for real-valued regression tasks.
        return np.real(predictions)


class QuantumKernelRegression2:
    """
    Implements a kernel regression model using quantum kernels.
    Supports SWAP/Trace kernels and finite measurement statistics (shot noise).
    """
    def __init__(self, regularization_lambda: float = 1e-6, num_shots: int = None):
        """
        Args:
            regularization_lambda: Adds to the diagonal to ensure invertibility.
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

    def _compute_raw_overlap(self, rho1: np.ndarray, rho2: np.ndarray) -> float:
        """Helper to compute exact Tr[rho1 * rho2]"""
        return np.real(np.trace(rho1 @ rho2))

    def _apply_physics_and_noise(self, exact_overlap: float) -> float:
        """
        Applies the correct physical measurement formula and binomial shot noise
        based on the chosen quantum circuit protocol.
        """
        # Exact overlap Tr[rho1 rho2] should strictly be between 0 and 1
        # Clipping handles tiny floating-point errors (e.g., 1.000000002)
        exact_overlap = np.clip(exact_overlap, 0.0, 1.0)
        
        # 1. Determine the exact theoretical probability of the measurement
        if self.kernel_type == "swap":
            # SWAP test measures an ancilla: P(0) = 0.5 * (1 + Tr)
            p_exact = 0.5 * (1.0 + exact_overlap)
            
        elif self.kernel_type in ["trace", "le", "loschmidt"]:
            # Loschmidt Echo measures the ground state: P(0..0) = Tr
            p_exact = exact_overlap
            
        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")

        # 2. Apply Shot Noise
        if self.num_shots is None:
            # Theoretical limit (infinite shots)
            return p_exact
        else:
            # Finite shots (Binomial sampling simulating the circuit measurements)
            measured_successes = np.random.binomial(n=self.num_shots, p=p_exact)
            return measured_successes / self.num_shots

    def fit(self, density_matrices: Union[List[np.ndarray], np.ndarray], 
                  labels: Union[List[float], np.ndarray], 
                  kernel_type: str = "trace"):
        
        if density_matrices is None or len(density_matrices) == 0:
            raise ValueError("Training density matrices cannot be empty.")
        if len(density_matrices) != len(labels):
            raise ValueError("Number of density matrices must match number of labels.")

        self.train_density_matrices = density_matrices
        self.train_labels = np.asarray(labels)
        self.kernel_type = kernel_type.lower() 

        n_samples = len(self.train_density_matrices)
        gram_matrix = np.zeros((n_samples, n_samples), dtype=float)

        for i in range(n_samples):
            # We measure each pair (i, j) only once to save shots and ensure the 
            # resulting estimated matrix is perfectly symmetric!
            for j in range(i, n_samples): 
                exact_overlap = self._compute_raw_overlap(self.train_density_matrices[i],
                                                          self.train_density_matrices[j])
                
                # Apply physics + noise
                val = self._apply_physics_and_noise(exact_overlap)
                
                gram_matrix[i, j] = val
                if i != j:
                    gram_matrix[j, i] = val

        # Regularization is EXTRA important with shot noise
        gram_matrix_reg = gram_matrix + self.regularization_lambda * np.eye(n_samples)

        try:
            self.K_inv = np.linalg.inv(gram_matrix_reg)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Gram matrix could not be inverted. Error: {e}")

        self.alpha = self.K_inv @ self.train_labels

    def predict(self, new_density_matrices: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        
        if self.K_inv.size == 0 or self.alpha.size == 0:
            raise RuntimeError("Model has not been fitted yet.")
        if new_density_matrices is None or len(new_density_matrices) == 0:
            return np.array([])

        n_test_samples = len(new_density_matrices)
        n_train_samples = len(self.train_density_matrices)
        
        kernel_test_train = np.zeros((n_test_samples, n_train_samples), dtype=float)

        for m in range(n_test_samples):
            for i in range(n_train_samples):
                exact_overlap = self._compute_raw_overlap(new_density_matrices[m],
                                                          self.train_density_matrices[i])
                
                kernel_test_train[m, i] = self._apply_physics_and_noise(exact_overlap)

        return kernel_test_train @ self.alpha