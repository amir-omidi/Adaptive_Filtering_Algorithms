import numpy as np
import matplotlib.pyplot as plt

# Generate a noisy sine wave signal
np.random.seed(0)
t = np.linspace(0, 1, 1000)
original_signal = np.sin(2 * np.pi * 5 * t)
noise = 0.5 * np.random.normal(0, 1, len(t))
noisy_signal = original_signal + noise

# LMS and NLMS parameters
filter_order = 20
mu_lms = 0.01
mu_nlms = 0.01

# Initialize filter weights for LMS and NLMS
w_lms = np.zeros(filter_order)
w_nlms = np.zeros(filter_order)
nlms_epsilon = 1e-3

# Store the recovered signals
recovered_signal_lms = []
recovered_signal_nlms = []

# Apply LMS and NLMS filtering
for i in range(filter_order, len(t)):  # Start from filter_order to avoid index errors
    x = noisy_signal[i:i - filter_order:-1]
    
    # LMS update
    e_lms = original_signal[i] - np.dot(w_lms, x)
    w_lms += 2 * mu_lms * e_lms * x
    recovered_signal_lms.append(np.dot(w_lms, x))
    
    # NLMS update
    e_nlms = original_signal[i] - np.dot(w_nlms, x)
    w_nlms += (2 * mu_nlms * e_nlms * x) / (nlms_epsilon + np.dot(x, x))
    recovered_signal_nlms.append(np.dot(w_nlms, x))

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title("Original vs. Noisy Signal")
plt.plot(t, original_signal, label="Original Signal", linewidth=2)
plt.plot(t, noisy_signal, label="Noisy Signal", alpha=0.7)
plt.legend()

plt.subplot(2, 1, 2)
plt.title("LMS vs. NLMS Filtered Signal")
plt.plot(t[filter_order:], original_signal[filter_order:], label="Original Signal", linewidth=2)
plt.plot(t[filter_order:], recovered_signal_lms, label="LMS Filtered Signal")
plt.plot(t[filter_order:], recovered_signal_nlms, label="NLMS Filtered Signal")
plt.legend()

plt.tight_layout()
plt.show()
