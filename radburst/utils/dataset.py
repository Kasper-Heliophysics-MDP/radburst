import os
import radburst.utils.utils as utils
import radburst.utils.preprocessing as prep

class Dataset:
    """Dataset class to manage loading, storing and processing data."""
    
    def __init__(self, data_dir):
        """Intialize the dataset.
        
        Args:
            data_dir (str): The root directory containing the FITS data files.

        Attributes:
            data_dir (str): The directory path for the dataset.
            data (list): List that stores the loaded data arrays from FITS files.
        """
        self.data_dir = data_dir
        self.data = []
        self.load_data()

    def load_data(self):
        """Load dataset from data_dir into data attribute.

        The data is loaded as a list of arrays, where each array corresponds to a 
        FITS file's raw spectrogram data.
        """
        for dirpath, dirnames, filenames in os.walk(self.data_dir):
            for file_name in filenames:
                file_path = os.path.join(dirpath, file_name)
                try:
                    spect_arr = utils.load_fits_file(file_path)
                    self.data.append(spect_arr)
                except Exception as e:
                    print(f'Error loading {file_path}: {e}')

        print(f'Loaded {len(self.data)} files from {self.data_dir}')


    def preprocess(self, standardize_rows = True, remove_vertical_lines = True):
        """Preprocess all elements in data according to arguments.
        
        Args:
            standardize_rows (bool): Standardize all rows (frequency channels) to reduce effects of horizontal noise.
            remove_vertical_lines (bool): Remove conistent (2-3) noisy vertical lines that appear in many collected LWA samples.          
        """
        for i, element in enumerate(self.data):
            if standardize_rows:
                element = prep.standardize_rows(element)
            if remove_vertical_lines:
                element = prep.remove_vertical_lines(element)
            self.data[i] = element


    def __getitem__(self, idx):
        """Return element at given index"""
        return self.data[idx]
    

    def __len__(self):
        """Return length of dataset"""
        return len(self.data)


    
    