import numpy as np
import matplotlib.pyplot as plt

def least_mean_squares(X, y, learning_rate=0.01):
    n_samples, n_features = X.shape
    theta = np.zeros((n_features, 1))
    theta_list = []

    for i in range(n_samples):
        x_i = X[i].reshape(-1, 1)
        prediction_error = y[i] - np.dot(x_i.T, theta)
        theta += learning_rate * prediction_error * x_i
        theta_list.append(theta.copy())

    return np.array(theta_list)

# Generate synthetic data
np.random.seed(0)
n_samples = 100
n_features = 2
X = np.random.rand(n_samples, n_features)
true_theta = np.array([[2], [3]])  # True parameters
y = np.dot(X, true_theta) + 0.5 * np.random.randn(n_samples, 1)  # Adding noise

# Apply LMS algorithm
theta_list = least_mean_squares(X, y)

# Plot the true parameter values and estimated parameter values over time
plt.figure(figsize=(10, 6))
plt.plot(theta_list[:, 0], label='Estimated theta[0]')
plt.plot(theta_list[:, 1], label='Estimated theta[1]')
plt.axhline(true_theta[0], color='r', linestyle='--', label='True theta[0]')
plt.axhline(true_theta[1], color='g', linestyle='--', label='True theta[1]')
plt.xlabel('Time')
plt.ylabel('Parameter Value')
plt.legend()
plt.title('Least Mean Squares (LMS) Algorithm')
plt.grid(True)
plt.show()
