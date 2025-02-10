import os

import skimage.transform
import utils.utils as utils
import utils.preprocessing as prep
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
import skimage
import numpy as np
import zipfile
from astropy.io import fits


class Dataset(TorchDataset):
    """Dataset class to manage loading, storing and processing data."""
    
    def __init__(self, data_dir, labels, preprocess=None, binary=True, zip=False):
        """Intialize the dataset.
        
        Args:
            data_dir (str): The root directory containing the FITS data files.
            labels (str or pd.DataFrame): Path to csv file containing labels (paths and burst types) or labels dataframe
            preprocess (callable, optional): Function that takes a spectrogram array and returns a preprocessed array.
                                             Defaults to None.
            binary (bool, optional): True for binary labels: 0 and 1 for no burst and burst
                                     False for type labels: burst number for burst, 0 for no burst
            zip (bool, optional):   True if data is extracted from multiple zip files
                                    False if data is extraced from folder on local hard drive

        Attributes:
            data_dir (str): The directory path for the dataset. 
            data (list): List that stores the loaded data arrays from FITS files.
        """
        self.data_dir = data_dir
        self.binary = binary
        self.preprocess = preprocess
        self.zip = zip

        # Load labels data
        if isinstance(labels, str):
            self.labels_df = pd.read_csv(labels)
        elif isinstance(labels, pd.DataFrame):
            self.labels_df = labels
        else:
            raise TypeError('labels must be a str path or a pd.DataFrame')
        
        self.paths = self.labels_df['path']
    

    # Function called whenever you enumerate the dataset
    def __getitem__(self, idx):
        '''
        Returns an item in dataset
        Called when you enumerate over dataloader
        
        '''

        # Load file
        file_path = os.path.join('FITfiles/', self.labels_df['path'].iloc[idx])
        spectrogram_arr = None
        file_found = False
        #if you want to use zipfile to avoid large local memory overhead
        if(self.zip):
            for zip_path in self.data_dir:
                zip_ref = zipfile.ZipFile(zip_path, 'r')
                fit_file_name = file_path
                if fit_file_name in zip_ref.namelist():
                    file_found = True
                    fit_file = zip_ref.open(fit_file_name)
                    fits_full_data = fits.open(fit_file)
                    spectrogram_arr = utils.load_fits_file(fits_full_data)
            if not file_found:
                print(f"error file: {file_path} not found")
                exit(1)


        #if you extracted the zip file in your local directory
        else:
            file_path = os.path.join(self.data_dir, self.labels_df['path'].iloc[idx])
            fits_full_data = fits.open(file_path)
            spectrogram_arr = utils.load_fits_file(fits_full_data)

        # Get label for file
        if self.binary:
            label = self.labels_df['burst'].iloc[idx]
        else:
            label = self.labels_df['type'].iloc[idx]
        # Preprocess
        if self.preprocess:
            spectrogram_arr = self.preprocess(spectrogram_arr)

        return {"spectrogram": spectrogram_arr, "label": label, "path": self.labels_df['path'].iloc[idx]}
    

    def __len__(self):
        '''
        Length of dataset
        '''
        return len(self.labels_df)


    def get_filtered_dataset(self, condition):
        '''
        Returns a new dataset
        This dataset will be a subset of the original dataset
        containing all points where the argument condition is true
        '''
        new_labels = self.labels_df.query(condition).reset_index(drop=True)
        new_dataset = Dataset(data_dir=self.data_dir,
                              labels=new_labels,
                              preprocess=self.preprocess,
                              binary=self.preprocess,
                              zip=self.zip)
        return new_dataset


    def only_bursts(self):
        return self.get_filtered_dataset(condition='burst == 1')
    

    def only_nonbursts(self):
        return self.get_filtered_dataset(condition='burst == 0')


class Resize():
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, array):
        return skimage.transform.resize(array, self.new_size)
    

class MinMaxNormalize():
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, array):
        """Add small epsilon to prevent division by zero"""
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val + self.eps)
