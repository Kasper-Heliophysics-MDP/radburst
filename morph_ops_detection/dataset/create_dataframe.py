import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset import Dataset
import utils.preprocessing as prep
import utils.candidates as cands
from torch.utils.data import DataLoader
import torch

data_path = ['data/FITfiles-20250205T173408Z-001.zip', 'data/FITfiles-20250205T173408Z-002.zip', 'data/FITfiles-20250205T173408Z-003.zip']
labels_path = 'data/metadata/filtered-labels-20240309-20240701.csv'

# Create a Dataset object which loads all data from the given path (defined in dataset.py)
data = Dataset(data_dir= data_path,
              labels= labels_path,
              zip=True)

dataloader = DataLoader(data, batch_size=1, shuffle=False)
bursts = []

from tqdm import tqdm

for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):   
    spect = batch['spectrogram'].numpy().squeeze()
    label = batch['label']
    # Create entry for dataframe
    reg_dict = cands.get_predicted_bbox(spect)

    # Add burst column value
    reg_dict['burst'] = int(label.item())

    bursts.append(reg_dict)

# Create dataframe for training
df = pd.DataFrame(bursts)
df.to_csv('bbox_data.csv', index=False)