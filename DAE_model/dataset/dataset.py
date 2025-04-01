import os
import sys
import skimage.transform
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
import skimage
import numpy as np
import zipfile
from astropy.io import fits
import requests
import gzip
from io import BytesIO
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import utils.utils as utils
import utils.preprocessing as prep


class Dataset(TorchDataset):
    """Dataset class to manage loading, storing and processing data."""
    
    def __init__(self, data_dir, labels, cache_folder=None, preprocess=None, binary=True, zip=False, resize=None, scaler=False, verbose=False):
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
        self.resize = resize
        self.scaler = scaler
        self.cache_folder = cache_folder
        self.verbose = verbose

        # Load labels data
        if isinstance(labels, str):
            self.labels_df = pd.read_csv(labels)
        elif isinstance(labels, pd.DataFrame):
            self.labels_df = labels
        else:
            raise TypeError('labels must be a str path or a pd.DataFrame')
        
        self.paths = self.labels_df['filename']
    

    # Function called whenever you enumerate the dataset
    def __getitem__(self, idx):
        '''
        Returns an item in dataset
        Called when you enumerate over dataloader
        
        '''

        #******************************
        # Get LWA data
        #******************************
        file_path = self.labels_df['filename'].iloc[idx]
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
            file_path = os.path.join(self.data_dir, self.labels_df['filename'].iloc[idx])
            fits_full_data = fits.open(file_path)
            spectrogram_arr = utils.load_fits_file(fits_full_data)


        #********************************
        # Get eCallisto data
        #********************************
        url = self.labels_df['link'].iloc[idx]
        file_name = url.split("/")[-1]
        spectrogram_arr_callisto = None
        if self.cache_folder: #if using a cache folder
            cached_file_path = os.path.join(self.cache_folder, file_name)
            if os.path.exists(cached_file_path): # Check if the file already exists in the cache
                if self.verbose:
                    print(f"Using cached file: {cached_file_path}") 
            else:
                if self.verbose:
                    print(f"Downloading: {url}") 

                response = requests.get(url, stream=True)

                if self.verbose:
                    response.raise_for_status()  # Raise an error if the download fails

                # Save the file to the cache folder
                with open(cached_file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                if self.verbose:
                    print(f"File downloaded and saved to: {cached_file_path}") 

            try:
                with gzip.open(cached_file_path, "rb") as gz_file:
                    decompressed_data = BytesIO(gz_file.read())
                    with fits.open(decompressed_data) as hdul:
                        if self.verbose:
                            hdul.info()  # Display FITS file structure
                        spectrogram_arr_callisto = hdul[0].data  # Access primary data (numpy array)
            except Exception as e:
                if self.verbose:
                    print(f"Downloading: {url}") 

                response = requests.get(url, stream=True)

                if self.verbose:
                    response.raise_for_status()  # Raise an error if the download fails

                with gzip.open(BytesIO(response.content), "rb") as gz_file:
                    decompressed_data = BytesIO(gz_file.read())
                    with fits.open(decompressed_data) as hdul:
                        if self.verbose:
                            hdul.info()  # Display FITS file structure
                        spectrogram_arr_callisto = hdul[0].data  # Access primary data (numpy array)
        else:
            response = requests.get(url, stream=True)

            if self.verbose:
                response.raise_for_status() 

            with gzip.open(BytesIO(response.content), "rb") as gz_file:
                decompressed_data = BytesIO(gz_file.read())
                with fits.open(decompressed_data) as hdul:
                    if self.verbose:
                        hdul.info()  # Display FITS file structure
                    spectrogram_arr_callisto = hdul[0].data  # Access primary data (numpy array)

        # Preprocess
        if self.preprocess:
            spectrogram_arr = self.preprocess(spectrogram_arr)
            spectrogram_arr_callisto = self.preprocess(spectrogram_arr_callisto)

        # Resize
        if self.resize:
            r = Resize(self.resize)
            spectrogram_arr = r(spectrogram_arr)
            spectrogram_arr_callisto = r(spectrogram_arr_callisto)

        # Scale
        if self.scaler:
            m = MinMaxNormalize()
            spectrogram_arr = m(spectrogram_arr)
            spectrogram_arr_callisto = m(spectrogram_arr_callisto)

        # Convert to tensor
        spectrogram_arr = torch.tensor(spectrogram_arr, dtype=torch.float32).unsqueeze(0)
        spectrogram_arr_callisto = torch.tensor(spectrogram_arr_callisto, dtype=torch.float32).unsqueeze(0)

        # Pytorch uses (batch, channels, height, width) but tensorflow uses (batch, height, width, channels)
        spectrogram_arr = spectrogram_arr.permute(1, 2, 0)
        spectrogram_arr_callisto = spectrogram_arr_callisto.permute(1, 2, 0)

        return {"peach_mountain_spectrogram": spectrogram_arr, "callisto_spectrogram": spectrogram_arr_callisto, "path": self.labels_df['filename'].iloc[idx], "datetime": self.labels_df['datetime'].iloc[idx], "label": self.labels_df['burst_type'].iloc[idx]}
    

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
                              cache_folder=self.cache_folder,
                              preprocess=self.preprocess,
                              binary=self.preprocess,
                              zip=self.zip,
                              resize=self.resize,
                              scaler=self.scaler,
                              verbose=self.verbose)
        return new_dataset


    def trainset(self):
        return self.get_filtered_dataset(condition='split == "train"')
    
    def validset(self):
        return self.get_filtered_dataset(condition='split == "valid"')
    
    def testset(self):
        return self.get_filtered_dataset(condition='split == "test"')
    
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
