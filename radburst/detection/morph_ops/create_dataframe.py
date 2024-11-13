import radburst.detection.morph_ops.candidates as cands
import os
from radburst.utils.dataset import Dataset
import radburst.utils.preprocessing as prep
import pandas as pd


data_path = '/mnt/c/Users/camer/OneDrive/Documents/radburst/data/Fitfiles'
labels_path = '/mnt/c/Users/camer/OneDrive/Documents/radburst/data/labels/filtered-labels-20240309-20240701.csv'

# Create a Dataset object which loads all data from the given path (defined in dataset.py)
data = Dataset(data_dir= data_path,
              labels= labels_path,
              preprocess= prep.stan_rows_remove_verts)

bursts = []

from tqdm import tqdm

for i, path_from_data_dir in tqdm(enumerate(data.paths), desc="Processing bursts", total=len(data.paths)):
    _, label = data[i]
    full_fits_path = os.path.join(data_path, path_from_data_dir)

    # Create entry for dataframe
    reg_dict = cands.get_predicted_bbox(fits_path=full_fits_path)

    # Add burst column value
    reg_dict['burst'] = int(label.item())

    bursts.append(reg_dict)

# Create dataframe for training
df = pd.DataFrame(bursts)
df.to_csv('bbox_data2.csv', index=False)