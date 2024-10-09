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