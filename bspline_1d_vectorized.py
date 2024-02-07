import numpy as np
import matplotlib.pyplot as plt

def cubic_b_spline_vectorized(x):
    """
    Vectorized calculation of cubic B-spline basis function values for a given x array.
    x can contain any real values, and the function will apply the cubic B-spline formula accordingly.
    """
    # B-spline basis function is defined piecewise
    abs_x = np.abs(x)
    cond1 = (abs_x >= 0) & (abs_x < 1)
    cond2 = (abs_x >= 1) & (abs_x < 2)
    
    # Calculate B-spline values under each condition
    result = np.zeros_like(x)
    result[cond1] = 2/3 - 0.5 * abs_x[cond1]**2 * (2 - abs_x[cond1])
    result[cond2] = 1/6 * (2 - abs_x[cond2])**3
    return result

def resize_signal_1d_vectorized(signal, new_length):
    """
    Resize a 1D signal to a new length using direct cubic B-spline interpolation in a vectorized form.
    """
    original_length = len(signal)
    
    # Compute the scaled indices of the new signal
    x_new = np.linspace(0, original_length - 1, new_length)
    
    # Compute the distances of each new index from all original indices
    x_dist = np.abs(np.subtract.outer(x_new, np.arange(original_length)))
    
    # Apply the cubic B-spline vectorized over all distances
    weights = cubic_b_spline_vectorized(x_dist)
    
    # Compute the weighted sum to get interpolated values
    resized_signal = np.dot(weights, signal)
    
    return resized_signal

# Example usage
original_signal = np.sin(np.linspace(0, 2 * np.pi, 10))  # Original signal
new_length = 50  # Desired length of the resized signal

resized_signal = resize_signal_1d_vectorized(original_signal, new_length)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, 2 * np.pi, 10), original_signal, 'o-', label='Original Signal')
plt.plot(np.linspace(0, 2 * np.pi, new_length), resized_signal, '.-', label='Resized Signal using Vectorized B-spline')
plt.legend()
plt.title("1D Signal Resizing with Vectorized B-spline Interpolation")
plt.show()