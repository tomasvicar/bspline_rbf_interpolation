import numpy as np
import matplotlib.pyplot as plt

def cubic_b_spline(t):
    """
    Calculate cubic B-spline basis function values for a given t in [0, 1].
    """
    if t >= 0 and t < 1:
        return 2/3 - 1/2 * t**2 * (2 - t)
    elif t >= 1 and t <= 2:
        return 1/6 * (2 - t)**3
    return 0

def resize_signal_1d(signal, new_length):
    """
    Resize a 1D signal using direct cubic B-spline interpolation.
    """
    # Original signal length and new signal setup
    original_length = len(signal)
    resized_signal = np.zeros(new_length)
    
    # Calculate the scale factor between the original and new signal
    scale_factor = (original_length - 1) / (new_length - 1)
    
    # Iterate over the new signal's indices to compute interpolated values
    for i in range(new_length):
        # Map the new index to the original signal's domain
        x = i * scale_factor
        
        # Summation over the B-spline basis functions multiplied by signal values
        s = 0
        for j in range(original_length):
            # Calculate the relative position in the original signal
            t = abs(x - j)
            s += cubic_b_spline(t) * signal[j if j < original_length else original_length - 1]
        
        resized_signal[i] = s
    
    return resized_signal

# Example usage
original_signal = np.sin(np.linspace(0, 2 * np.pi, 10))  # Original signal
new_length = 50  # Desired length of the resized signal

resized_signal = resize_signal_1d(original_signal, new_length)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, 2 * np.pi, 10), original_signal, 'o-', label='Original Signal')
plt.plot(np.linspace(0, 2 * np.pi, new_length), resized_signal, '.-', label='Resized Signal')
plt.legend()
plt.show()
