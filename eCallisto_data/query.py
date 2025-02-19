import sys
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from website_access import website as web
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.preprocessing as prep
import utils.utils as util
from utils.dataset import Dataset

url = "https://soleil.i4ds.ch/solarradio/data/BurstLists/2010-yyyy_Monstein/2024/"

# Just testing if the link works
all_files = web.list_files("https://soleil.i4ds.ch/solarradio/data/BurstLists/2010-yyyy_Monstein/2024/", ".txt")
print("List of all files:")
print(all_files)

#init datasets
data_path = ['C:\\Users\\15862\\OneDrive\\Documents\\Senior year assignments\\helio\\data\\FITfiles-20250205T173408Z-001.zip', 'C:\\Users\\15862\\OneDrive\\Documents\\Senior year assignments\\helio\\data\\FITfiles-20250205T173408Z-002.zip', 'C:\\Users\\15862\\OneDrive\\Documents\\Senior year assignments\\helio\\data\\FITfiles-20250205T173408Z-003.zip']
labels_path = 'C:\\Users\\15862\\OneDrive\\Documents\\Senior year assignments\\helio\\data\\metadata\\filtered-labels-20240309-20240701.csv'
data = Dataset(data_dir= data_path,
              labels= labels_path,
              zip=True)

dataset_only_bursts = data.only_bursts()
dataset_only_nonbursts = data.only_nonbursts()
df = pd.DataFrame(columns=['matched_bursts', 'unmatched_nonbursts'])
burst_batch_dict = {}
nonburst_batch_dict = {}


# Bursts
print("Burst data")
burst_dataloader = DataLoader(dataset_only_bursts, batch_size=1, shuffle=False)

curr_month = "00"
callisto_data = {}
for batch_idx, batch in tqdm(enumerate(burst_dataloader), total=len(burst_dataloader)): 
    batch_datetime_est = batch['datetime'][0]
    batch_datetime_utc = web.convert_to_utc(batch_datetime_est)
    batch_time = batch_datetime_utc[11:]
    batch_month = batch_datetime_utc[5:7]
    batch_day = batch_datetime_utc[8:10]
    found = False

    # If current loaded file is from a different month than data
    if batch_month != curr_month:
        curr_month = batch_month
        search_date = batch_datetime_utc[0:7].replace("-", "_")

        # Load file and make dictionary
        response = web.get_file(url, search_date)
        print("Text found on file: ")
        print(response.text)
        callisto_data = web.read_burst_list(response)

    for index, entry in callisto_data.iterrows():

        # If this eCallisto burst happened within the date and time range that our station was recording
        if web.is_within_range(entry['time'], web.get_15_range(batch_time)) and batch_month == entry['date'][4:6] and batch_day == entry['date'][6:8]:
            found = True
            burst_batch_dict[batch_datetime_est] = ", ".join(entry['stations'])
            # Loop over stations that observed this boost
            for station in entry['stations']:

                # Add to df if we haven't seen this station yet
                if not station in df.index:
                    df.loc[station] = [0, 0]

                # Burst in our data matches burst in eCallisto data
                df.loc[station, 'matched_bursts'] = df.loc[station, 'matched_bursts'] + 1

    if not found:
        burst_batch_dict[batch_datetime_est] = []

print(df)

# Nonbursts
print("Nonburst Data")
nonburst_dataloader = DataLoader(dataset_only_nonbursts, batch_size=1, shuffle=False)

curr_month = "00"
callisto_data = {}
for batch_idx, batch in tqdm(enumerate(nonburst_dataloader), total=len(nonburst_dataloader)): 
    batch_datetime_est = batch['datetime'][0]
    batch_datetime_utc = web.convert_to_utc(batch_datetime_est)
    batch_time = batch_datetime_utc[11:]
    batch_month = batch_datetime_utc[6:7]
    found = False

    # If current loaded file is from a different month than data
    if batch_month != curr_month:
        curr_month = batch_month
        search_date = batch_datetime_utc[0:7].replace("-", "_")

        # Load file and make dictionary
        response = web.get_file(url, search_date)
        print("Text found on file: ")
        print(response.text)
        callisto_data = web.read_burst_list(response)

    for index, entry in callisto_data.iterrows():

        # If this eCallisto burst happened within the date and time range that our station was recording
        if web.is_within_range(entry['time'], web.get_15_range(batch_time)) and batch_month == entry['date'][4:6] and batch_day == entry['date'][6:8]:

            found = True
            burst_batch_dict[batch_datetime_est] = ", ".join(entry['stations'])
            # Loop over stations that observed this boost
            for station in entry['stations']:

                # Add to df if we haven't seen this station yet
                if not station in df.index:
                    df.loc[station] = [0, 0]

                # NonBurst in our data matches burst in eCallisto data
                df.loc[station, 'unmatched_nonbursts'] = df.loc[station, 'unmatched_nonbursts'] + 1

    if not found:
        nonburst_batch_dict[batch_datetime_est] = []

print(df)
df.to_csv('eCallisto_station_matches.csv', index=True)

csv_filename = "burst_matching_stations.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["batch_datetime", "stations"])
    for datetime_key, stations in burst_batch_dict.items():
        writer.writerow([datetime_key, stations])  

csv_filename = "nonburst_but_matching_stations.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["batch_datetime", "stations"])
    for datetime_key, stations in nonburst_batch_dict.items():
        writer.writerow([datetime_key, stations])  








    
