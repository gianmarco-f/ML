
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

    