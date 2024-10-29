import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage import label
from skimage.measure import regionprops
import cv2


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


def stan_rows_remove_verts(arr):
    """Standardize rows and remove vertical lines"""
    res = standardize_rows(arr)
    res = remove_vertical_lines(res)
    return res


class BinaryMaskRegion:
    def __init__(self, bbox):
        self.min_row, self.min_col, self.max_row, self.max_col = bbox
        self.height = self.max_row - self.min_row
        self.width = self.max_col - self.min_col
        self.hw_ratio = self.height / self.width
        self.area = self.height * self.width


class RegionManager:
    def __init__(self):
        self.regions = []

    def add_region(self, region):
        self.regions.append(region)

    def _get_largest_2_regions(self):
        if len(self.regions) < 2: return self.regions
        sorted_area = sorted(self.regions, key=lambda r: r.area)
        return sorted_area[-2:]
    
    def filter_largest_2_regions(self, row_diff_threshold=50, size_ratio_threshold=10):
        if len(self.regions) < 2: return self.regions

        # sometimes mask images have one large low (high row) region and one small high (low row) region 
        # higest row = low frequency

        largest_2_regions = self._get_largest_2_regions()
        largest_reg = largest_2_regions[1]
        second_largest_reg = largest_2_regions[0]

        max_row_diff = largest_reg.max_row - second_largest_reg.max_row
        size_ratio = largest_reg.area / second_largest_reg.area        

        # if difference in highest row is greater than threshold, keep region with greater max row (lower in image)
        if abs(max_row_diff) > row_diff_threshold:
            return [largest_reg] if max_row_diff > 0 else [second_largest_reg]

        # if the largest is greater than some factor larger than the second largest, only keep largest
        if size_ratio > size_ratio_threshold:
            return [largest_reg]
        
        return largest_2_regions
  

def create_binary_mask(arr, pct_threshold = 95):
    threshold = np.percentile(arr, pct_threshold)
    return arr > threshold


def morph_ops(arr, erosion_struct_size = (10,3), dilation_struct_size = (1,5)):
    """Make array binary and perform morphological operations to remove small components."""
    arr = binary_dilation(arr, structure=np.ones((3,20)))

    eroded_mask = binary_erosion(arr, structure=np.ones(erosion_struct_size))
    dilation_mask = binary_dilation(eroded_mask, structure=np.ones(dilation_struct_size))
    return dilation_mask


def filtered_components(mask, 
                        min_hw_ratio = 0.2, 
                        min_area = 600, 
                        min_h_wide_tall = 30, 
                        min_area_wide_tall = 1000):
    """Only keep connected components (regions) in binary mask that meet criteria."""
    
    filtered_mask = np.zeros_like(mask)
    region_manager = RegionManager()

    labeled_mask, _ = label(mask)
    properties = regionprops(labeled_mask)

    for prop in properties:
        reg = BinaryMaskRegion(bbox=prop.bbox)

        # Two criteria for keeping regions 
            # main criteria for minimum height/width ratio and minimum area
            # second criteria for bursts that don't pass main ratio but might be type 2 or 5
        main_criteria = reg.hw_ratio >= min_hw_ratio and reg.area > min_area
        keep_wide_tall_bursts_criteria = reg.height > min_h_wide_tall and reg.area > min_area_wide_tall
    
        # If the region meets criteria insert into manager
        if main_criteria or keep_wide_tall_bursts_criteria:
            filtered_mask[labeled_mask == prop.label] = 1
            region_manager.add_region(region=reg)

    # out of all regions added to manager (that passed two criteria in loop above)
    # we will take the largest 2 regions (if there are >=2) and check two additional criteria
    # the additional criteria are size_ratio and max_row_diff, more info in RegionManager class
    filtered_largest_2_regions = region_manager.filter_largest_2_regions()

    return filtered_largest_2_regions, filtered_mask


def blur(arr, blur_filter_shape = (51,11)):
    return cv2.GaussianBlur(arr,blur_filter_shape,0)  
