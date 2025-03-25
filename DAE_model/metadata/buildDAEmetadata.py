import zipfile
import pandas as pd
import sys
import os
import random
from astropy.io import fits
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import utils.utils as utils
import utils.time_utils as clock
import utils.website as web

def get_random_split():
    choices = ["valid", "train", "test"]
    weights = [0.2, 0.6, 0.2]  # Adjust these weights as needed
    return random.choices(choices, weights=weights, k=1)[0]

df = pd.DataFrame()

zips = ['data/FITfiles-20250205T173408Z-001.zip', 'data/FITfiles-20250205T173408Z-002.zip', 'data/FITfiles-20250205T173408Z-003.zip']
Callisto_fit_url = "http://soleil80.cs.technik.fhnw.ch/solarradio/data/2002-20yy_Callisto/"
labels_csv = 'data/metadata/classification_labels_raw.csv'
labels_df = pd.read_csv(labels_csv)
station = "ALASKA"
output_csv = 'DAE_model/metadata/DAEmetadata.csv'

if(0):
    for zip_path in zips:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            for file_name in zipf.namelist():
                            
                # Search eCallisto website for matching fit file
                timestamp = clock.extract_timestamp(file_name)
                timestamp = clock.round_to_nearest_15(timestamp)

                basename = os.path.basename(file_name)  # Extracts "Beelink1_20241022_183000_59.fit"
                basename_without_ext = os.path.splitext(basename)[0] # Removes the .fit

                # Search labels csv for burst info
                match = labels_df[labels_df.iloc[:, 0].str.contains(basename_without_ext, na=False, case=False)]

                burst = 0
                burst_type = 0
                if not match.empty:
                    burst = 1
                    burst_type = match.iloc[0, 1]

                new_row = pd.DataFrame([{'filename': file_name, 'datetime': timestamp, 'burst': burst, 'burst_type': burst_type, 'link': " ", "split": get_random_split()}])
                df = pd.concat([df, new_row], ignore_index=True)

    print(df)
    df['datetime'] = pd.to_datetime(df['datetime'], format="%Y%m%d_%H:%M:%S")

    df = df.sort_values(by='datetime')
    df['datetime'] = df['datetime'].dt.strftime("%Y-%m-%d %H:%M:%S")
    print("Sorted df by date")

    df.to_csv(output_csv)

if(0):
    df = pd.read_csv(output_csv)

    current_date = None
    file_list = None
    for index, row in df.iterrows():
        utc = clock.est_to_utc(row['datetime'])
        new_date = clock.time_helper(utc)['date']
        new_time = clock.time_helper(utc)['time']
        if new_date != current_date:
            current_date = new_date
            current_datetime = utc
            url = Callisto_fit_url + clock.time_helper(current_datetime)['year'] + "/" + clock.time_helper(current_datetime)['month'] + "/" + clock.time_helper(current_datetime)['day'] + "/"
            file_list = web.list_files(url)

        search_date = new_date.replace("-", "")
        search_time = new_time.replace(":", "")
        
        matching_dates = [f for f in file_list if search_date in f]
        matching_dates_and_times = [f for f in matching_dates if search_time in f]
        alaska_file = next((f for f in matching_dates_and_times if "ALASKA" in f), None)
        if alaska_file:
            df.at[index, 'link'] = alaska_file
            print(alaska_file)
        else:
            print(f"File not found: {alaska_file}")

    df.to_csv(output_csv)

if(1):
    df = pd.read_csv(output_csv)

    # Initialize lists to store the spectrogram sizes
    spectrogram_t = []
    spectrogram_f = []

    for _, row in df.iterrows():
        file_path = row['filename']  
        file_found = False

        # Search for the file in the ZIP archives
        for zip_path in zips:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                if file_path in zip_ref.namelist():
                    file_found = True
                    with zip_ref.open(file_path) as fit_file:
                        try:
                            with fits.open(fit_file, ignore_missing_simple=True) as fits_full_data:
                                spectrogram_arr = utils.load_fits_file(fits_full_data)
                                spectrogram_t.append(spectrogram_arr.shape[1])
                                spectrogram_f.append(spectrogram_arr.shape[0])
                        except OSError as e:
                            spectrogram_t.append(None)
                            spectrogram_f.append(None)
                            print(f"Error processing {file_path} in {zip_path}: {e}")
                    break

        if not file_found:
            # Placeholder if file not found
            print(f"Error: File '{file_path}' not found in the ZIP archives.")
            spectrogram_t.append(None)  
            spectrogram_f.append(None)

    # Add column to dataframe and print to csv
    df['spectrogram_f'] = spectrogram_f
    df['spectrogram_t'] = spectrogram_t
    

    df.to_csv(output_csv, index=False)