
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
        if not density_matrices:
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
        if not new_density_matrices:
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

# --- 4. Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    num_train_samples = 10
    num_test_samples = 3
    qubit_dimension = 2 # e.g., for a single qubit system, density matrix is 2x2
    # Adjust regularization_lambda if you encounter issues with inversion
    # A smaller value means less regularization, closer to the original K.
    # A larger value increases stability but might bias the solution.
    REG_LAMBDA = 1e-6 

    print("--- Generating Example Data ---")
    # Generate random training density matrices and labels
    train_rhos = [generate_random_density_matrix(qubit_dimension) for _ in range(num_train_samples)]
    # Example labels: simple binary labels for demonstration
    train_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]) # Or continuous values
    # For a regression problem, labels would typically be continuous values.
    # For binary classification, you might apply a sign function or threshold later.

    # Generate random test density matrices
    test_rhos = [generate_random_density_matrix(qubit_dimension) for _ in range(num_test_samples)]

    print("\n--- Initializing and Fitting Model ---")
    # Initialize the model
    model = QuantumKernelRegression(regularization_lambda=REG_LAMBDA)

    # Fit the model to the training data
    model.fit(train_rhos, train_labels)

    print(f"\nTraining complete. Calculated alpha (dual variables):\n{model.alpha}")

    print("\n--- Making Predictions ---")
    # Make predictions on new (test) density matrices
    predictions = model.predict(test_rhos)

    print("\n--- Results ---")
    print(f"Test Density Matrices (first one shown):\n{test_rhos[0]}")
    print(f"Predicted values for test density matrices:\n{predictions}")

    # You can also compute predictions for the training data itself to check
    # how well the model learned the training set (should be close to train_labels)
    train_predictions = model.predict(train_rhos)
    print(f"\nPredictions on training data (first 5):\n{train_predictions[:5]}")
    print(f"Original training labels (first 5):\n{train_labels[:5]}")
    
    # Calculate Mean Squared Error on training data
    mse = np.mean((train_predictions - train_labels)**2)
    print(f"Mean Squared Error on training data: {mse:.4f}")
    