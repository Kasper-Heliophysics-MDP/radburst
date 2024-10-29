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
            labels (str or pd.DataFrame): Path to csv file containing labels (paths and burst types) or labels dataframe
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

        # Load labels data
        if isinstance(labels, str):
            self.labels_df = pd.read_csv(labels)
        elif isinstance(labels, pd.DataFrame):
            self.labels_df = labels
        else:
            raise TypeError('labels must be a str path or a pd.DataFrame')
        
        self.paths = self.labels_df['path']
    

    def __getitem__(self, idx):

        # Load file
        file_path = os.path.join(self.data_dir, self.labels_df['path'].iloc[idx])
        spectrogram_arr = utils.load_fits_file(file_path)

        # Get label for file
        if self.binary:
            label = self.labels_df['burst'].iloc[idx]
        else:
            label = self.labels_df['type'].iloc[idx]

        # Preprocss
        if self.preprocess:
            spectrogram_arr = self.preprocess(spectrogram_arr)

        return spectrogram_arr, label
    

    def __len__(self):
        return len(self.labels_df)


    def get_filtered_dataset(self, condition):
        new_labels = self.labels_df.query(condition).reset_index(drop=True)
        new_dataset = Dataset(data_dir=self.data_dir,
                              labels=new_labels,
                              preprocess=self.preprocess,
                              binary=self.preprocess)
        return new_dataset


    def only_bursts(self):
        return self.get_filtered_dataset(condition='burst == 1')
    

    def only_nonbursts(self):
        return self.get_filtered_dataset(condition='burst == 0')
