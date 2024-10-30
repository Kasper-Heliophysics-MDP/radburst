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
        np.ndarray: Processed array with vertical lines removed.
    """
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
    """Standardize rows and remove vertical lines from spectrogram array.
    
    Args:
        arr (np.ndarray): Array to process.
        
    Returns:
        np.ndarray: Processed array - standardized rows and vertical lines removed.
    """
    res = standardize_rows(arr)
    res = remove_vertical_lines(res)
    return res


class BinaryMaskRegion:
    """Represents a bounding box (bbox) for a connected component in a binary mask.
    
    Attributes:
        min_row (int): The minimum row index of the bounding box.
        min_col (int): The minimum column index of the bounding box.
        max_row (int): The maximum row index of the bounding box.
        max_col (int): The maximum column index of the bounding box.
        height (int): The height of the bounding box.
        width (int): The width of the bounding box.
        hw_ratio (float): The height-to-width ratio of the bounding box.
        area (int): The area of the bounding box.

    Args:
        bbox (tuple): A tuple containing the bounding box coordinates
                      in the format (min_row, min_col, max_row, max_col).
    """
    def __init__(self, bbox):
        self.min_row, self.min_col, self.max_row, self.max_col = bbox
        self.height = self.max_row - self.min_row
        self.width = self.max_col - self.min_col
        self.hw_ratio = self.height / self.width
        self.area = self.height * self.width


class RegionManager:
    """Manages a collection of regions and provides methods to analyze/filter them.
    
    This class allows adding regions, finding the two largest regions and filtering 
    based on specific criteria such as row difference and size ratio.
    
    Attributes:
        regions (list): A list of `BinaryMaskRegion` instances that represent the regions.  
    """
    def __init__(self):
        self.regions = []

    def add_region(self, region):
        self.regions.append(region)

    def _get_largest_2_regions(self):
        """Return up to two of the largest regions."""
        if len(self.regions) < 2: return self.regions
        sorted_area = sorted(self.regions, key=lambda r: r.area)
        return sorted_area[-2:]
    
    def filter_largest_2_regions(self, row_diff_threshold=50, size_ratio_threshold=10):
        """Filters the two largest regions based on criteria.

        Checks the difference in max row indices and area ratio of the largest regions
        to determine which regions to keep.
        
        Args:
            row_diff_threshold (int):
            size_ratio_threshold (int):
            
        Returns:
            list: A list containing the filtered `BinaryMaskRegion` instances.
        """
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
    """Create a binary mask of an array.
    
    Args:
        arr (np.ndarray): Array to create mask.
        pct_threshold (int): Percentile threshold for setting mask values to 1.
                             For example, if this value is 95, the top 95% of array 
                             values will be 1, the rest 0.
        
    Returns:
        np.ndarray: Binary mask where values above threshold are 1 and others are 0
    """
    threshold = np.percentile(arr, pct_threshold)
    return arr > threshold


def morph_ops(binary_arr, 
              dilation_1_struct_size = (3,20),
              erosion_struct_size = (10,3), 
              dilation_2_struct_size = (1,5)):
    """Apply morphological operations to a binary mask to refine components/structures.

    The goal is to remove small structures and enhance larger structures (potential bursts).
    
    Args:
        binary_arr (np.ndarray): Input binary array (mask) to be processed.
        dilation_1_struct_size (tuple): Structure element size for first dilation step.
        erosion_struct_size (tuple): Structure element size for erosion step.
        dilation_2_struct_size (tuple): Structure element size for second dilation step.
        
    Returns:
        np.ndarray: Refined binary mask with morphological operations applied.
    """
    dilation_1_mask = binary_dilation(binary_arr, structure=np.ones(dilation_1_struct_size))
    eroded_mask = binary_erosion(dilation_1_mask, structure=np.ones(erosion_struct_size))
    dilation_2_mask = binary_dilation(eroded_mask, structure=np.ones(dilation_2_struct_size))
    return dilation_2_mask


def filtered_components(mask, 
                        min_hw_ratio = 0.2, 
                        min_area = 600, 
                        min_h_wide_tall = 30, 
                        min_area_wide_tall = 1000):
    """Filter connected components (regions) in a binary mask based on size and shape criteria.

    First, filter out regions by two sets of criteria (main and secondary) that evaluate
    height-to-width ratio and area. Then, keep upto the two largest remaining regions and 
    apply two additional criteria from the RegionManager class: area ratio and max row difference to
    reduce false positive detections of a second burst.

    The secondary criteria (`keep_wide_tall_bursts`) was added to include type 2 bursts, which
    the main criteria exclude due to the restrictive height-to-width ratio.

    Args:
        mask (np.ndarray): Binary mask of connected components to filter.
        min_hw_ratio (float): Minimum height-to-width ratio for main filter criteria.
        min_area (int): Minimum area for main filter criteria.
        min_h_wide_tall (int): Minimum height-to-widt ratio for keep_wide_tall_bursts filter criteria.
        min_area_wide_tall (int): Minimum area for keep_wide_tall_bursts filter criteria.

    Returns:
        tuple: A tuple containing:
            - filtered_largest_largest_2_regions (list): List of BinaryMaskRegion objects - regions that passed 
                                                         additional criteria found in RegionManager class. 
            - filtered_mask (np.ndarray): Mask with only regions that passed main and secondary criteria.
    """
    
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
    """Blur array using a Gaussian kernel to enhance potential bursts.

    Args:
        arr (np.ndarray): Input array to blur.
        blur_filter_shape (tuple): Gaussian kernel size (width, height).

    Returns:
        np.ndarray: Blurred array.
    """
    return cv2.GaussianBlur(arr,blur_filter_shape,0)  
