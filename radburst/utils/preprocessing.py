import numpy as np

def standardize_rows(arr):
    """Standardize each row by subtracting the mean and dividing by the standard deviation.

    Args:
        arr (np.ndarray): Array to standardize.

    Returns:
        np.ndarray: Row-standardized 2D array.
    """
    mean_per_row = np.mean(arr, axis=1, keepdims=True)
    std_per_row = np.std(arr, axis=1, keepdims=True)

    # Prevent divide by 0 errors
    epsilon = 1e-8
    std_per_row = np.maximum(std_per_row, epsilon)

    standardized_arr = (arr - mean_per_row) / std_per_row

    return standardized_arr


def remove_vertical_lines(arr, num_std = 5, dist = 10):
    """Remove columns with high variance (vertical lines) with other columns some distance away.

    The default value of 5 seems to be a value based on a few random samples but this could be further verified. 
    For the dist, we can see that these vertical lines are always just a few columns so 10 is far enough away to 
    get a replacement column.
    
    Args:
        arr (np.ndarray): Array to process.
        num_std (int): Number of standard deviations above the mean to use as threshold for vertical lines.
        dist (int): Distance away from vertical line column to get replacement column.
        
    Returns:
        np.ndarray: Processed array with vertical lines removed."""
    
    # Calculate variance of each column
    vars = np.var(arr, axis=0, ddof=0)

    # Calculate the mean and standard deviation of the variances
    mean_var = np.mean(vars)
    std_var = np.std(vars)

    # Calculate the threshold for detecting columns that contain unwanted vertical liens
    var_threshold = mean_var + num_std * std_var

    # Find indices of columns with variance greater than threshold
    high_var_cols = np.where(vars > var_threshold)[0]

    # Replace high variance columns with other columns some distance away
        # We know the vertical lines are only a few cols wide
    arr_verts_removed = arr.copy()
    for i in high_var_cols:
        col_to_replace_vert_line = (i - dist) if (i >= dist) else (i + dist)
        arr_verts_removed[:,i] = arr[:,col_to_replace_vert_line]
        
    return arr_verts_removed


def softmax(arr):
    """Apply the Softmax function to an input array.

    The Softmax function normalizes an array by converting values into a probability distribution.
    
    Args:
        arr (np.ndarray): Array
    
    Returns:
        np.ndarray: Normalized array with same shape as input and values in [0,1].
    """
    max_val = np.max(arr, axis=0)
    e_x = np.exp(arr - max_val)
    return e_x / np.sum(e_x, axis=0)


def standardize_rows(arr):
    """Standardize each row by subtracting the mean and dividing by the standard deviation.

    Args:
        arr (np.ndarray): Array to standardize.

    Returns:
        np.ndarray: Row-standardized 2D array.
    """
    mean_per_row = np.mean(arr, axis=1, keepdims=True)
    std_per_row = np.std(arr, axis=1, keepdims=True)

    # Prevent divide by 0 errors
    epsilon = 1e-8
    std_per_row = np.maximum(std_per_row, epsilon)

    standardized_arr = (arr - mean_per_row) / std_per_row

    return standardized_arr


def min_max_norm(arr):
    """Scale values in input array to range [0,1]

    Args:
        arr (np.ndarray): 1D array to normalize.

    Returns:
        np.ndarray: 1D normalized array.
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def resize_interp(arr, new_size):
    """Resize a 1D array using interpolation.
    
    Args:
        arr (np.ndarray): 1D array to resize.
    
    Returns:
        nd.ndarray: 1D resized array.
    """
    original_indices = np.arange(len(arr))
    new_indices = np.linspace(0, len(arr) - 1, new_size)
    return np.interp(new_indices, original_indices, arr)


def col_expected_vals(arr, norm=True):
    """Calculate the expected value of each column"""
    
    # y data (column index for now but could be mapped to freq to get expected freq)
    col_index = np.arange(arr.shape[0]).reshape(1,-1)

    # Make each col a probability distribution (weights for sum)
    soft_max_cols = softmax(arr) 
    
    # Weighted sum of each col
    col_expected_vals = np.sum(np.dot(col_index, soft_max_cols), axis=0)

    return min_max_norm(col_expected_vals) if norm else col_expected_vals


def col_sums(arr, norm=True):
    """Calculate the sum of each column (timestep) across all frequencies"""

    sums = np.sum(arr, axis=0)
    return min_max_norm(sums) if norm else sums


def stan_rows_remove_verts(arr):
    """Standardize rows and remove vertical lines"""
    res = standardize_rows(arr)
    res = remove_vertical_lines(res)
    return res