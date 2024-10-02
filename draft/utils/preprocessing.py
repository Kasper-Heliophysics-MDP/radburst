import numpy as np

def standardize_rows(arr):
    """
    Standardize each row by subtracting the mean and dividing by the standard deviation.

    Args:
        arr (np.ndarray): 2D array to standardize.

    Returns:
        np.ndarray: Row-standardized 2D array.
    """
    mean_per_row = np.mean(arr, axis=1, keepdims=True)
    std_per_row = np.std(arr, axis=1, keepdims=True)

    # Prevent divide by 0 errors
    epsilon = 1e-8
    std_per_row = np.maximum(std_per_row, epsilon)

    return (arr - mean_per_row) / std_per_row