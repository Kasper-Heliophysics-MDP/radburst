import os
import radburst.utils.utils as utils
import radburst.utils.preprocessing as prep
import pandas as pd

class Dataset:
    """Dataset class to manage loading, storing and processing data."""
    
    def __init__(self, data_dir, labels, preprocess=None, binary=True):
        """Intialize the dataset.
        
        Args:
            data_dir (str): The root directory containing the FITS data files.
            labels (str): Path to csv file containing labels (paths and burst types)
            preprocess (callable, optional): Function that takes a spectrogram array and returns a preprocessed array.
                                             Defaults to None.
            binary (bool, optional): True for binary labels: 0 and 1 for no burst and burst
                                     False for type labels: burst number for burst, 0 for no burst

        Attributes:
            data_dir (str): The directory path for the dataset.
            data (list): List that stores the loaded data arrays from FITS files.
        """
        self.data_dir = data_dir
        self.binary = binary
        self.preprocess = preprocess
        
        if isinstance(labels, str):
            self.labels_df = pd.read_csv(labels)
        elif isinstance(labels, pd.DataFrame):
            self.labels_df = labels
        else:
            raise ValueError('Labels must be a file path or pandas dataframe.')

        self.file_path_col = self.labels_df['path']
        self.burst_type_col = self.labels_df['type']
        self.binary_label_col = self.labels_df['burst']
    

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_path_col.iloc[idx])

        # Load file
        spectrogram_arr = utils.load_fits_file(file_path)

        # Get label for file: 
            # for binary labels are in {0,1} for burst or non-burst
            # otherwise labels are in {2,3,4,5,6,7} for the burst type
        if self.binary:
            label = self.binary_label_col.iloc[idx]
        else:
            label = self.burst_type_col.iloc[idx]

        # Preprocss
        if self.preprocess:
            spectrogram_arr = self.preprocess(spectrogram_arr)

        return spectrogram_arr, label
    

    def __len__(self):
        return len(self.labels_df)


    def only_bursts(self):
        return Dataset(data_dir=self.data_dir,
                       labels=self.labels_df[self.labels_df['burst'] == 1],
                       preprocess=self.preprocess,
                       binary=self.binary)
    

    def only_nonbursts(self):
        return Dataset(data_dir=self.data_dir,
                       labels=self.labels_df[self.labels_df['burst'] == 0],
                       preprocess=self.preprocess,
                       binary=self.binary)
