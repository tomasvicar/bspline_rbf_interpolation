import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

from numpy.fft import fftn, ifftn, fftshift

import numpy as np

def bilinear_interpolation(image, coordinates):

    x, y = coordinates[..., 1], coordinates[..., 0]
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    x0_clip = np.clip(x0, 0, image.shape[1] - 1)
    x1_clip = np.clip(x1, 0, image.shape[1] - 1)
    y0_clip = np.clip(y0, 0, image.shape[0] - 1)
    y1_clip = np.clip(y1, 0, image.shape[0] - 1)

    Ia = image[y0_clip, x0_clip, ...]
    Ib = image[y1_clip, x0_clip, ...]
    Ic = image[y0_clip, x1_clip, ...]
    Id = image[y1_clip, x1_clip, ...]

    wa = ((x1 - x) * (y1 - y))
    wb = ((x1 - x) * (y - y0))
    wc = ((x - x0) * (y1 - y))
    wd = ((x - x0) * (y - y0))

    if image.ndim == 3:
        interpolated_values = wa[..., np.newaxis] * Ia + wb[..., np.newaxis] * Ib + wc[..., np.newaxis] * Ic + wd[..., np.newaxis] * Id
    else:
        interpolated_values = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return interpolated_values


def fft_convolve2d(img, kernel):
    """Convolve an image with a kernel using FFT in the spectral domain."""
    # Pad the image and kernel to avoid circular convolution effects
    padded_img = np.pad(img, kernel.shape[0] // 2, mode='reflect')
    
    # Perform FFT on both the image and the kernel
    img_fft = fftn(padded_img, axes=(0, 1))
    kernel_fft = fftn(kernel, s=padded_img.shape, axes=(0, 1))
    
    # Element-wise multiplication in the spectral domain
    result_fft = img_fft * kernel_fft
    
    # Inverse FFT to transform the result back to the spatial domain
    result = ifftn(result_fft, axes=(0, 1)).real
    
    # Crop the result back to the original image size
    result = result[kernel.shape[0]:, kernel.shape[1]:]
    
    return result

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


def cubic_b_spline_kernel(size):
    """
    Generate a 1D cubic B-spline kernel based on the size.
    """
    x = np.linspace(-2, 2, size)
    kernel = cubic_b_spline_vectorized(x)
    # kernel /= np.sum(kernel)  # Normalize the kernel
    return kernel

def resize_grid_2d_convolution(grid, new_shape):
    """
    Resize a 2D grid by inserting zeros and then applying convolution with a cubic B-spline kernel.
    """
    original_shape = grid.shape
    scale_y, scale_x = new_shape[0] / original_shape[0], new_shape[1] / original_shape[1]

    scale_y_round = np.ceil(scale_y)
    scale_x_round = np.ceil(scale_x)

    new_shape_round = (int(scale_y_round * original_shape[0]), int(scale_x_round * original_shape[1]))


    padded_grid = np.zeros(new_shape_round)
    padded_grid[int(scale_y_round // 2)::int(scale_y_round), int(scale_x_round // 2)::int(scale_x_round)] = grid

    kernel_size_y = int(3 * scale_y_round) + 3
    kernel_size_x = int(3 * scale_x_round) + 3
    
    kernel_y = cubic_b_spline_kernel(kernel_size_y + (1 - kernel_size_y % 2)) 
    kernel_x = cubic_b_spline_kernel(kernel_size_x + (1 - kernel_size_y % 2))


    kernel = np.outer(kernel_y, kernel_x) 

    resized_grid = fft_convolve2d(padded_grid, kernel)

    amplitude_scale_factor = (scale_y_round * scale_x_round) / kernel.sum() 
    resized_grid *= amplitude_scale_factor

    positions_final = np.meshgrid(np.linspace(0, new_shape_round[0] - 1, new_shape[0]), np.linspace(0, new_shape_round[1] - 1, new_shape[1]), indexing='ij')
    positions_final = np.stack(positions_final, axis=-1)
    resized_grid_final = bilinear_interpolation(resized_grid, positions_final)

    return resized_grid_final 

# Example usage: Create a 2D signal (e.g., a simple gradient)
original_grid = np.outer(np.sin(np.linspace(0, 2 * np.pi, 10)), np.cos(np.linspace(0, 2 * np.pi, 10)))

# Desired shape of the resized grid
new_shape = (52, 52)

resized_grid_convolution = resize_grid_2d_convolution(original_grid, new_shape)

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(original_grid, interpolation='nearest')
plt.title("Original Grid")

plt.subplot(1, 2, 2)
plt.imshow(resized_grid_convolution, interpolation='nearest')
plt.title("Resized Grid using Convolution with Cubic B-spline")
plt.show()



# Selecting a slice for quality comparison
original_slice = original_grid[:, 5]  # Take a column slice from the middle
resized_slice = resized_grid_convolution[:, int(5 * new_shape[1] / original_grid.shape[1])]

# Plotting the slices for quality comparison
plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, original_grid.shape[0], original_grid.shape[0]), original_slice, label="Original Slice", marker='o')
plt.plot(np.linspace(0, original_grid.shape[0], resized_grid_convolution.shape[0]), resized_slice, label="Resized Slice", linestyle='-', linewidth=1)
plt.legend()
plt.title("Comparison of Original vs. Resized Grid Slice")
plt.xlabel("Index")
plt.ylabel("Signal Value")
plt.show()

print(np.std(original_grid))
print(np.std(resized_grid_convolution))