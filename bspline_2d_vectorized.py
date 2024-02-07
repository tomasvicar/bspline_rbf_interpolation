import numpy as np
import matplotlib.pyplot as plt

def cubic_b_spline_vectorized(x):
    """
    Vectorized calculation of cubic B-spline basis function values for a given x array.
    """
    abs_x = np.abs(x)
    cond1 = (abs_x >= 0) & (abs_x < 1)
    cond2 = (abs_x >= 1) & (abs_x < 2)
    
    result = np.zeros_like(x)
    result[cond1] = 2/3 - 0.5 * abs_x[cond1]**2 * (2 - abs_x[cond1])
    result[cond2] = 1/6 * (2 - abs_x[cond2])**3
    return result

def resize_grid_2d(grid, new_shape):
    """
    Resize a 2D grid using direct cubic B-spline interpolation in a vectorized form.
    """
    original_shape = grid.shape
    
    # Compute the scaled indices for the new grid
    y_new = np.linspace(0, original_shape[0] - 1, new_shape[0])
    x_new = np.linspace(0, original_shape[1] - 1, new_shape[1])
    
    # Compute distances for each dimension
    y_dist = np.abs(np.subtract.outer(y_new, np.arange(original_shape[0])))
    x_dist = np.abs(np.subtract.outer(x_new, np.arange(original_shape[1])))
    
    # Apply cubic B-spline vectorized over all distances
    weights_y = cubic_b_spline_vectorized(y_dist)
    weights_x = cubic_b_spline_vectorized(x_dist)
    
    # Interpolate in y-dimension
    intermediate_grid = np.dot(weights_y, grid)
    
    # Interpolate in x-dimension
    resized_grid = np.dot(intermediate_grid, weights_x.T)
    
    return resized_grid

# Example usage: Create a 2D signal (e.g., a simple gradient)
original_grid = np.outer(np.sin(np.linspace(0, 2 * np.pi, 10)), np.cos(np.linspace(0, 2 * np.pi, 10)))

# Desired shape of the resized grid
new_shape = (50, 50)

resized_grid = resize_grid_2d(original_grid, new_shape)

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(original_grid, interpolation='nearest')
plt.title("Original Grid")

plt.subplot(1, 2, 2)
plt.imshow(resized_grid, interpolation='nearest')
plt.title("Resized Grid using Vectorized B-spline")
plt.show()

