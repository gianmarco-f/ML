import time
import numpy as np
import matplotlib.pyplot as plt
# import myQML

####### Helper Function #########
def get_n_qubit_observable(n, base_op):
    """
    Generates an n-qubit observable by taking the tensor product 
    of the base_op n times. E.g., for base_op = X, returns X^otimes n.
    """
    obs = base_op
    for _ in range(n - 1):
        obs = np.kron(obs, base_op)
    return obs

####### Settings #########
n_qubits_list = [1, 2, 3] # The number of qubits (n)
N_train = 100                
N_test = 200                 
num_realizations = 30
num_shots = 1000

# CRITICAL: Since num_shots=1000, we MUST use regularization to prevent matrix inversion explosion!
reg_lambda = 0 

# QELM reservoir physical setup (Fixed part)
d_res = 64
num_povm_elements = 20

# Generate POVM ONCE since it only depends on the fixed d_res
povm = myQML.generate_random_povm(d_res, num_povm_elements)

# Storage dictionaries (Updated for quantiles)
results_mean = {'QELM': [], 'Kernel_LE': [], 'Kernel_SWAP': []}
results_median = {'QELM': [], 'Kernel_LE': [], 'Kernel_SWAP': []}
quantiles_5 = {'QELM': [], 'Kernel_LE': [], 'Kernel_SWAP': []}
quantiles_95 = {'QELM': [], 'Kernel_LE': [], 'Kernel_SWAP': []}

print(f"-------- (Fixed N_train = {N_train}, Shots = {num_shots}, Realizations = {num_realizations}) --------")
print("="*70)

total_time = time.time()

for n in n_qubits_list:
    d_in = 2**n              
    d_out = d_in * d_res     
    
    print(f"Evaluating n = {n} qubits (d_in = {d_in})...")
    mse_runs = {'QELM': [], 'Kernel_LE': [], 'Kernel_SWAP': []}
    step_time = time.time()

    obs_n = get_n_qubit_observable(n, myQML.X)

    for r in range(num_realizations):
        # Using vectorization for massive speedup
        ds = myQML.QuantumDatasetGenerator(N_train, N_test, obs_n)
        ds.generate_density_matrices_vec()
        ds.compute_expectation_values_vec()
        
        rho_train, y_train = ds.get_training_dataset()
        rho_test, y_test = ds.get_test_dataset()

        # QELM
        V = myQML.random_isometry(d_in, d_out)
        qelm = myQML.QuantumExtremeLearningMachine(
            isometry=V, 
            povm=povm, 
            bipartite_dims=(d_in, d_res), 
            keep_subsystem=1, 
            num_shots=num_shots
        )
        qelm.fit_vec(rho_train, y_train)
        mse_runs['QELM'].append(np.mean((qelm.predict_vec(rho_test) - y_test)**2))
        
        # Kernel LE
        kernel_le = myQML.QuantumKernelRegression(reg_lambda, num_shots)
        kernel_le.fit_vec(rho_train, y_train, "le")
        mse_runs['Kernel_LE'].append(np.mean((kernel_le.predict_vec(rho_test) - y_test)**2))

        # Kernel SWAP
        kernel_swap = myQML.QuantumKernelRegression(reg_lambda, num_shots)
        kernel_swap.fit_vec(rho_train, y_train, "swap")
        mse_runs['Kernel_SWAP'].append(np.mean((kernel_swap.predict_vec(rho_test) - y_test)**2))

    # Store results (Mean, Median, 5% and 95% Quantiles)
    for key in results_mean.keys():
        results_mean[key].append(np.mean(mse_runs[key]))
        results_median[key].append(np.median(mse_runs[key]))
        quantiles_5[key].append(np.percentile(mse_runs[key], 5))
        quantiles_95[key].append(np.percentile(mse_runs[key], 95))

    print(f"  -> QELM MSE: {results_mean['QELM'][-1]:.4f}")
    print(f"  -> K-LE MSE: {results_mean['Kernel_LE'][-1]:.4f}")
    print(f"  -> K-SW MSE: {results_mean['Kernel_SWAP'][-1]:.4f}")
    print(f"  -> Time:     {time.time() - step_time:.1f}s")

print("="*70)
print(f"Experiment finished in {(time.time() - total_time)/60:.1f} minutes.")

# ==========================================
# Plotting
# ==========================================
plt.figure(figsize=(11, 7))
n_arr = np.array(n_qubits_list)

colors = {'QELM': 'purple', 'Kernel_LE': 'blue', 'Kernel_SWAP': 'red'}
labels = {'QELM': f'QELM ($d_{{res}}={d_res}$, $n_{{povm}}={num_povm_elements}$)', 
          'Kernel_LE': 'Kernel (LE)', 
          'Kernel_SWAP': 'Kernel (SWAP)'}
markers = {'QELM': 'o', 'Kernel_LE': 's', 'Kernel_SWAP': '^'}

for model in results_mean.keys():
    mean_arr = np.array(results_mean[model])
    median_arr = np.array(results_median[model])
    q5_arr = np.array(quantiles_5[model])
    q95_arr = np.array(quantiles_95[model])
    
    # Plot Median (Solid)
    plt.plot(n_arr, median_arr, label=f'{labels[model]} (Median)', color=colors[model], 
             marker=markers[model], linestyle='-', linewidth=2.5, markersize=8)
    
    # Plot Mean (Dashed, Empty marker)
    plt.plot(n_arr, mean_arr, label=f'{labels[model]} (Mean)', color=colors[model], 
             marker=markers[model], linestyle='--', linewidth=2.0, markersize=8, fillstyle='none')
    
    # Fill between 5% and 95% quantiles
    plt.fill_between(n_arr, q5_arr, q95_arr, color=colors[model], alpha=0.15)

plt.xticks(n_qubits_list) 
plt.yscale('log') 

plt.xlabel('Number of Qubits ($n$)', fontsize=14)
plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
plt.title(f'Scaling with System Size: QELM vs Quantum Kernels\n(Fixed $N_{{train}} = {N_train}$, {num_shots} Shots)', fontsize=16)

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=11, ncol=2, loc='upper left')
plt.tight_layout()
plt.show()