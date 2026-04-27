import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Tuple
from scipy.linalg import sqrtm, inv

#definition on Pauli matrices
I = np.array([[1, 0],
              [0, 1]], dtype=complex)

X = np.array([[0, 1],
              [1, 0]], dtype=complex)

Y = np.array([[0, -1j],
              [1j,  0]], dtype=complex)

Z = np.array([[1,  0],
              [0, -1]], dtype=complex)

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

def HS_inner_product(rho1, rho2):
    """
    Computes the Hilbert-Schmidt inner product between two density matrices rho1 and rho2.
    """
    #matrix multiplication and trace
    return np.trace(rho1 @ rho2)

def partial_trace(rho, dims: Tuple[int,int], keep):
    """
    Fast pure-Numpy implementation of partial trace using Einstein summation.
    
    Args:
        rho: The input density matrix (2D array).
        dims: A list of dimensions of the subsystems (e.g., [2, 32]).
        keep: The index of the subsystem to keep (0 or 1).

    Returns the reduced density matrix after tracing out the other subsystem.
    """
    rho_tensor = rho.reshape(dims[0], dims[1], dims[0], dims[1])
    if keep == 0:
        # Keep subsystem 0, trace out subsystem 1
        reduced_rho = np.einsum('ijkj->ik', rho_tensor)
    elif keep == 1: 
        # Keep subsystem 1, trace out subsystem 0
        reduced_rho = np.einsum('ijil->jl', rho_tensor)
    else:
        raise ValueError("Invalid 'keep' index. Must be 0 or 1.")
    return reduced_rho


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



    # =================================== #
    # Old function for dataset generation #
    # =================================== #

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
