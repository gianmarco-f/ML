import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Tuple
from scipy.linalg import sqrtm, inv

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def random_density_matrix(dim):
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    A = (A + A.conj().T) / 2
    A = A @ A.conj().T
    return A / np.trace(A)

# --- NEW OPTIMIZED FUNCTION ---
def random_density_matrices(dim, N):
    """Generates N random density matrices of dimension dim simultaneously."""
    A = np.random.randn(N, dim, dim) + 1j * np.random.randn(N, dim, dim)
    A = (A + A.conj().transpose(0, 2, 1)) / 2
    rho = A @ A.conj().transpose(0, 2, 1)
    # Fast vectorized trace and normalization
    traces = np.trace(rho, axis1=1, axis2=2)[:, np.newaxis, np.newaxis]
    return rho / traces

def HS_inner_product(rho1, rho2):
    return np.trace(rho1 @ rho2)

def partial_trace(rho, dims: Tuple[int,int], keep):
    rho_tensor = rho.reshape(dims[0], dims[1], dims[0], dims[1])
    if keep == 0:
        return np.einsum('ijkj->ik', rho_tensor)
    elif keep == 1: 
        return np.einsum('ijil->jl', rho_tensor)
    else:
        raise ValueError("Invalid 'keep' index.")

def random_isometry(d_in: int, d_out: int) -> np.ndarray:
    X = (np.random.randn(d_out, d_out) + 1j * np.random.randn(d_out, d_out)) / np.sqrt(2)
    Q, R = np.linalg.qr(X)
    diag = np.diag(R)
    phases = diag / np.abs(diag)
    Q = Q * phases
    return Q[:, :d_in]

def generate_random_povm(dim: int, num_elements: int) -> List[np.ndarray]:
    elements = []
    total_sum = np.zeros((dim, dim), dtype=complex)
    for _ in range(num_elements):
        A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        M = A @ A.conj().T 
        elements.append(M)
        total_sum += M
    normalization = inv(sqrtm(total_sum))
    return [normalization @ E @ normalization for E in elements]

def generate_computational_povm(dim: int) -> List[np.ndarray]:
    povm = []
    for i in range(dim):
        E = np.zeros((dim, dim), dtype=complex)
        E[i, i] = 1.0
        povm.append(E)
    return povm

def density_to_bloch(rho):
    x = np.real(np.trace(rho @ X))
    y = np.real(np.trace(rho @ Y))
    z = np.real(np.trace(rho @ Z))
    return np.array([x, y, z])

def plot_bloch_sphere(points):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.1)
    ax.quiver(0,0,0, 1,0,0, color='r')
    ax.quiver(0,0,0, 0,1,0, color='g')
    ax.quiver(0,0,0, 0,0,1, color='b')
    points = np.array(points)
    ax.scatter(points[:,0], points[:,1], points[:,2], color='black', s=40)
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])
    plt.show()

def generate_dataset_pauli(N_training, N_test, pauli_matrix):
    rho_training = random_density_matrices(2, N_training)
    rho_test = random_density_matrices(2, N_test)
    expectation_training = np.real(np.einsum('nij,ji->n', rho_training, pauli_matrix))
    expectation_test = np.real(np.einsum('nij,ji->n', rho_test, pauli_matrix))
    return rho_training, rho_test, expectation_training, expectation_test