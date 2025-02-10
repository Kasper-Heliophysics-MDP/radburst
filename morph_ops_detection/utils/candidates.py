#from preprocessing import stan_rows_remove_verts, blur, create_binary_mask, morph_ops, filtered_components
import utils.preprocessing as prep
import utils.utils as util
import numpy as np


def get_lowest_freq_bbox(reg_manager):
    """Get the highest frequency bbox (region object with higest max_row"""
    regions = reg_manager.regions
    if len(regions) == 0: return None  

    sorted_regions = sorted(regions, key=lambda r: (-r.max_row, -r.area))
    return sorted_regions[0]    


def get_candidates(raw):
    """"""
    stan_rows = prep.stan_rows_remove_verts(raw)
    stan_rows = np.clip(stan_rows, 0, 5)
    stan_rows_blur = prep.blur(stan_rows)
    binary_mask = prep.create_binary_mask(stan_rows_blur, 95)
    eroded_mask = prep.morph_ops(binary_mask)    
    _, _, region_manager = prep.filtered_components(eroded_mask)

    return region_manager, stan_rows


def get_predicted_bbox(spect):   
    """"""   
    reg_man, stan_rows = get_candidates(spect)                      

    reg = get_lowest_freq_bbox(reg_man)
    
    if reg:   
        # bbox found
        height = reg.max_row - reg.min_row
        width = reg.max_col - reg.min_col
        area = height*width
        inside_box = stan_rows[reg.min_row:reg.max_row+1, reg.min_col:reg.max_col+1]

        return {'height': reg.height,
                'width': reg.width,
                'area': reg.area,
                'max_row': reg.max_row,
                'min_row': reg.min_row,
                'mean': np.mean(inside_box),
                'median': np.median(inside_box),
                'hw_ratio': reg.hw_ratio}

    else:
        return {'height': 0,
                'width': 0,
                'area': 0,
                'max_row': 0,
                'min_row': 0,
                'mean': 0,
                'median': 0,
                'hw_ratio': 0}