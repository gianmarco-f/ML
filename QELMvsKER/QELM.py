import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#let's define the Pauli matrices
I = np.array([[1, 0],
              [0, 1]], dtype=complex)

X = np.array([[0, 1],
              [1, 0]], dtype=complex)

Y = np.array([[0, -1j],
              [1j,  0]], dtype=complex)

Z = np.array([[1,  0],
              [0, -1]], dtype=complex)

N_tr = 100
N_test = 1000

#density matrices
rho_training = np.zeros((N_tr, 2, 2), dtype=complex)
rho_test = np.zeros((N_test,2,2), dtype=complex)

#expectation values
expe_X_train = np.zeros(N_tr)
expe_X_test = np.zeros(N_test)

expe_Y_train = np.zeros(N_tr)
expe_Y_test = np.zeros(N_test)

expe_Z_train = np.zeros(N_tr)
expe_Z_test = np.zeros(N_test)

for i in range(N_tr):
    #sample a random Bloch vector uniformly in the Bloch ball
    r = np.random.normal(size=3)
    r = r / np.linalg.norm(r) * np.random.rand()**(1/3)
    
    #construct density matrix
    rho_training[i] = 0.5 * (I + r[0]*X + r[1]*Y + r[2]*Z)

    #expectation value of Pauli X
    expe_X_train[i] = np.real(np.trace(rho_training[i] @ X))
    expe_Y_train[i] = np.real(np.trace(rho_training[i] @ Y))
    expe_Z_train[i] = np.real(np.trace(rho_training[i] @ Z))

for i in range(N_test):
    #sample a random Bloch vector uniformly in the Bloch ball
    r = np.random.normal(size=3)
    r = r / np.linalg.norm(r) * np.random.rand()**(1/3)
    
    #construct density matrix
    rho_test[i] = 0.5 * (I + r[0]*X + r[1]*Y + r[2]*Z)

    #cxpectation value of Pauli X
    expe_X_test[i] = np.real(np.trace(rho_test[i] @ X))
    expe_Y_test[i] = np.real(np.trace(rho_test[i] @ Y))
    expe_Z_test[i] = np.real(np.trace(rho_test[i] @ Z))


def generate_random_density_matrix(dim):
    # 1. Create a random complex matrix G
    G = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim))
    
    # 2. Compute G * G_dagger (makes it Hermitian and Positive)
    A = G @ G.conj().T
    
    # 3. Normalize the trace to 1
    rho = A / np.trace(A)
    return rho

def random_isometry(d_in=2, d_out=64):
    """
    A function to generate a random isometry
    """
    X = (np.random.randn(d_out, d_out) +
         1j * np.random.randn(d_out, d_out)) / np.sqrt(2)
    Q, R = np.linalg.qr(X)

    diag = np.diag(R)
    phases = diag / np.abs(diag)
    Q = Q * phases

    return Q[:, :d_in]

V = random_isometry(2, 64)
print(np.allclose(V.conj().T @ V, np.eye(2)))

def generate_random_povm(dim, num_elements):
    """ 
    A function to generate a random povm
    """
    elements = []
    total_sum = np.zeros((dim, dim), dtype=complex)
    
    for _ in range(num_elements):
        # Generate a random complex matrix
        A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        # Make it positive semi-definite (A * A_dagger)
        M = A @ A.conj().T
        elements.append(M)
        total_sum += M
        
    # Normalize so they sum to Identity
    # Use the matrix square root of the inverse sum to normalize
    from scipy.linalg import sqrtm, inv
    normalization = inv(sqrtm(total_sum))
    
    povm = [normalization @ E @ normalization for E in elements]
    return povm


counts = list(range(1, 17))


povm_families = {n: generate_random_povm(32,n) for n in counts}
    
c = sum(povm_families[5])

print(c)

def out_probabilites(rho, povm, shots):
    """
    Computes the estimated probabilities od povm outcomes based
    on a finite/infinte number of measurements.
    """
    # if shots = 0, we return the theoretical probabilities without sampling
    if shots == 0:

        probs = [(M @ rho).trace().real for M in povm]

        probs = np.array(probs) / np.sum(probs)

        return probs
    
    else: 
         #calculate theoretical probabilities: p_i = Tr(M_i * rho)
         #we take the real part because Tr is Hermitian but might have tiny imag noise
         probs = [(M @ rho).trace().real for M in povm]
        
         #ensure probabilities sum to 1
         probs = np.array(probs) / np.sum(probs)
        
         #sample from the multinomial distribution
         counts = np.random.multinomial(shots, probs)
        
         #observed frequencies (Finite Statistics)
         observed_probs = counts / shots
        
         return observed_probs
    


rho_te= generate_random_density_matrix(32)
povm_te = povm_families[10]

est_probs = out_probabilites(rho_te, povm_te, 0)

print(est_probs)
print(np.sum(est_probs))


def generate_dataset_probs(rho_list, povm, shots, V):
    """
    Input: List of qutip density matrices
    Output: A 2D numpy array [num_states x num_povm_elements]
    """
    num_states = len(rho_list)
    num_elements = len(povm)
    prob_matrix = np.zeros((num_states, num_elements))
    
    for idx, rho in enumerate(rho_list):
        # Using the finite statistics logic from before

        # Evolution and measurement
        rho_evolved = partial_trace((V @ rho @ V.conj().T), 1, [2,32])
        est_probs = out_probabilites(rho_evolved, povm, shots=shots)
        prob_matrix[idx, :] = est_probs
        
    return prob_matrix.T


train_prob = {}
k = {}

for n, povm in povm_families.items():

    train_prob[n] = {}
    k[n] = {}

    shot_list = [0, 1000, 10000, 100000, 1000000]

    for s in shot_list:
        train_prob[n][s] = generate_dataset_probs(rho_training, povm, s, V)

        singular_values = np.linalg.svd(train_prob[n][s], compute_uv=False)
        if n < 4:
            k[n][s] = singular_values[0]/singular_values[-1]
        else:
        
            k[n][s] = singular_values[0]/singular_values[3]


# 3. Create the plot
singular_test_1 = np.linalg.svd(train_prob[11][0],compute_uv=False)

print
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(singular_test_1) + 1), singular_test_1, 'o-', markersize=8, linewidth=2)

# Labeling
plt.title('Singular Values', fontsize=14)
plt.xlabel('Index (Rank)', fontsize=12)
plt.ylabel('Singular Value ($\sigma_i$)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Annotate the extremes
plt.annotate(f'Biggest: {singular_test_1[0]:.2f}', xy=(1, singular_test_1[0]), xytext=(3, singular_test_1[0]+0.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1))
plt.annotate(f'Smallest: {singular_test_1[-1]:.2f}', xy=(len(singular_test_1), singular_test_1[-1]), xytext=(len(singular_test_1)-6, singular_test_1[-1]+2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1))

plt.show()


# 3. Create the plot
singular_test_2 = np.linalg.svd(train_prob[12][1000] ,compute_uv=False)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(singular_test_2) + 1), singular_test_2, 'o-', markersize=8, linewidth=2)

# Labeling
plt.title('Singular Values', fontsize=14)
plt.xlabel('Index (Rank)', fontsize=12)
plt.ylabel('Singular Value ($\sigma_i$)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Annotate the extremes
plt.annotate(f'Biggest: {singular_test_2[0]:.2f}', xy=(1, singular_test_2[0]), xytext=(3, singular_test_2[0]+0.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1))
plt.annotate(f'Smallest: {singular_test_2[-1]:.2f}', xy=(len(singular_test_2), singular_test_2[-1]), xytext=(len(singular_test_2)-6, singular_test_2[-1]+2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1))

plt.show()

n_values = sorted(k.keys())
shot_values = sorted(k[n_values[0]].keys())

plt.figure(figsize=(10, 6))

# 2. Loop through shots to create one line per 's'
for s in shot_values:
    # Extract MSE for this specific 's' across all 'n'
    y_values = [k[n][s] for n in n_values]
    
    label = f"Shots: {s}" if s > 0 else "Exact ($s=\infty$)"
    plt.plot(n_values, y_values, marker='o', linestyle='-', label=label)

# 3. Formatting
plt.yscale('log') # Usually necessary for MSE to see the scale of improvement
plt.xlabel('POVM Family ($n$)')


plt.title('Condition number')
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()


w_trained = {}

for n, povm in povm_families.items():
    # Initialize a sub-dictionary for this specific 'n'
    w_trained[n] = {}
    
    # Define the shot counts you want to test
    shot_list = [0, 1000, 10000, 100000, 1000000]
    
    for s in shot_list:
        # Store using the shot count as the key

        # pseudo = np.linalg.pinv(generate_dataset_probs(rho_training, povm, s, V))
        w_trained[n][s] = np.reshape(expe_X_train,(-1,1)).T @ (np.linalg.pinv(train_prob[n][s]))

test_prob = {}

for n, povm in povm_families.items():

    test_prob[n] = {}

    shot_list = [0, 1000, 10000, 100000, 1000000]

    for s in shot_list:
        test_prob[n][s] = generate_dataset_probs(rho_test, povm, s, V)



MSE_Ntrain_infty = {}
for n in povm_families.keys():
    MSE_Ntrain_infty[n] = {}

    shot_list = [1000, 10000, 100000, 1000000]

    for s in shot_list:
        w = w_trained[n][0]
        X_Ntrain_infty = w @ test_prob[n][s]
        MSE_Ntrain_infty[n][s] = np.mean((X_Ntrain_infty - np.reshape(expe_X_test, (-1,1)).T)**2)


n_values = sorted(MSE_Ntrain_infty.keys())
shot_values = sorted(MSE_Ntrain_infty[n_values[0]].keys())

plt.figure(figsize=(10, 6))

# 2. Loop through shots to create one line per 's'
for s in shot_values:
    # Extract MSE for this specific 's' across all 'n'
    y_values = [MSE_Ntrain_infty[n][s] for n in n_values]
    
    label = f"Shots: {s}" if s > 0 else "Exact ($s=\infty$)"
    plt.plot(n_values, y_values, marker='o', linestyle='-', label=label)

# 3. Formatting
plt.yscale('log') # Usually necessary for MSE to see the scale of improvement
plt.xlabel('POVM elements ($n$)')
plt.ylim(1e-6, 1e2)
plt.ylabel('Mean Squared Error')
plt.title('MSE($\infty$, Ntest)')
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()

MSE_Ntest_infty = {}
for n in povm_families.keys():
    MSE_Ntest_infty[n] = {}

    shot_list = [1000, 10000, 100000, 1000000]

    for s in shot_list:
        w = w_trained[n][s]
        X_Ntest_infty = w @ test_prob[n][0]
        MSE_Ntest_infty[n][s] = np.mean((X_Ntest_infty - np.reshape(expe_X_test, (-1,1)).T)**2)

n_values = sorted(MSE_Ntest_infty.keys())
shot_values = sorted(MSE_Ntest_infty[n_values[0]].keys())

plt.figure(figsize=(10, 6))

# 2. Loop through shots to create one line per 's'
for s in shot_values:
    # Extract MSE for this specific 's' across all 'n'
    y_values = [MSE_Ntest_infty[n][s] for n in n_values]
    
    label = f"Shots: {s}" if s > 0 else "Exact ($s=\infty$)"
    plt.plot(n_values, y_values, marker='o', linestyle='-', label=label)

# 3. Formatting
plt.yscale('log') # Usually necessary for MSE to see the scale of improvement
plt.xlabel('POVM elements ($n$)')
plt.ylim(1e-6, 1e2)
plt.ylabel('Mean Squared Error')
plt.title('MSE (Ntrain, $\infty$)')
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()


MSE_finite = {}
for n in povm_families.keys():
    MSE_finite[n] = {}

    shot_list = [1000, 10000, 100000, 1000000]

    for s in shot_list:
        w = w_trained[n][s]
        X_infty = w @ test_prob[n][s]
        MSE_finite[n][s] = np.mean((X_infty - np.reshape(expe_X_test, (-1,1)).T)**2)

n_values = sorted(MSE_finite.keys())
shot_values = sorted(MSE_finite[n_values[0]].keys())


plt.figure(figsize=(10, 6))

for s in shot_values:
    
    y_values = [MSE_finite[n][s] for n in n_values]
    
    label = f"Shots: {s}" if s > 0 else "Exact ($s=\infty$)"
    plt.plot(n_values, y_values, marker='o', linestyle='-', label=label)


plt.yscale('log')
plt.xlabel('POVM elements ($n$)')
plt.ylim(1e-6, 1e2)
plt.ylabel('Mean Squared Error')
plt.title('MSE(N train, Ntest)')
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()
