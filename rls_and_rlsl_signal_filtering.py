import numpy as np
import matplotlib.pyplot as plt

# Generate a noisy sine wave signal
np.random.seed(0)
t = np.linspace(0, 1, 1000)
original_signal = np.sin(2 * np.pi * 5 * t)
noise = 0.5 * np.random.normal(0, 1, len(t))
noisy_signal = original_signal + noise

# RLS and RLSL parameters
filter_order = 20
lambda_rls = 0.99  # For RLS
lambda_rlsl = 0.98  # For RLSL

# Initialize filter weights and covariance matrices for RLS and RLSL
w_rls = np.zeros(filter_order)
P_rls = np.eye(filter_order)
w_rlsl = np.zeros(filter_order)
P_rlsl = np.eye(filter_order)
rlsl_epsilon = 1e-3

# Store the recovered signals
recovered_signal_rls = []
recovered_signal_rlsl = []

# Apply RLS and RLSL filtering
for i in range(filter_order, len(t)):  # Start from filter_order to avoid index errors
    x = noisy_signal[i:i - filter_order:-1]
    
    # RLS update
    y_rls = np.dot(w_rls, x)
    e_rls = original_signal[i] - y_rls
    K_rls = (P_rls @ x) / (lambda_rls + np.dot(x, P_rls @ x))
    w_rls += K_rls * e_rls
    P_rls = (P_rls - np.outer(K_rls, x) @ P_rls) / lambda_rls
    recovered_signal_rls.append(y_rls)
    
    # RLSL update
    y_rlsl = np.dot(w_rlsl, x)
    e_rlsl = original_signal[i] - y_rlsl
    K_rlsl = (P_rlsl @ x) / (lambda_rlsl + np.dot(x, P_rlsl @ x))
    w_rlsl += K_rlsl * e_rlsl
    P_rlsl = (P_rlsl - np.outer(K_rlsl, x) @ P_rlsl) / lambda_rlsl + rlsl_epsilon * np.eye(filter_order)
    recovered_signal_rlsl.append(y_rlsl)

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title("Original vs. Noisy Signal")
plt.plot(t, original_signal, label="Original Signal", linewidth=2)
plt.plot(t, noisy_signal, label="Noisy Signal", alpha=0.7)
plt.legend()

plt.subplot(2, 1, 2)
plt.title("RLS vs. RLSL Filtered Signal")
plt.plot(t[filter_order:], original_signal[filter_order:], label="Original Signal", linewidth=2)
plt.plot(t[filter_order:], recovered_signal_rls, label="RLS Filtered Signal")
plt.plot(t[filter_order:], recovered_signal_rlsl, label="RLSL Filtered Signal")
plt.legend()

plt.tight_layout()
plt.show()
