import zipfile
import pandas as pd
import sys
import os
import random
from astropy.io import fits
import requests
import gzip
from io import BytesIO
import skimage
import numpy as np
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import utils.utils as utils
import utils.time_utils as clock
import utils.website as web

def get_random_split():
    choices = ["valid", "train", "test"]
    weights = [0.2, 0.6, 0.2]  # Adjust these weights as needed
    return random.choices(choices, weights=weights, k=1)[0]

def init_metadata(outcsv, zips, labelcsv):
    '''
    Initializes a metadata CSV file by extracting file information from ZIP archives
    and associating it with labeled burst data.

    Args:
        outcsv (str): Path to the output CSV file where metadata will be saved.
        zips (list of str): List of ZIP archive paths containing FITS files.
        labelcsv (str): Path to the CSV file containing burst labels.
    '''

    labels_df = pd.read_csv(labelcsv)
    for zip_path in zips:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            for file_name in zipf.namelist():
                            
                # Get the timestamp
                timestamp = clock.extract_timestamp(file_name)

                basename = os.path.basename(file_name)  
                basename_without_ext = os.path.splitext(basename)[0] 

                # Search labels csv for burst info
                match = labels_df[labels_df.iloc[:, 0].str.contains(basename_without_ext, na=False, case=False)]

                burst = 0
                burst_type = 0
                if not match.empty:
                    burst = 1
                    burst_type = match.iloc[0, 1]

                new_row = pd.DataFrame([{'filename': file_name, 'datetime': timestamp, 'burst': burst, 'burst_type': burst_type, "initial_mse": None, 'link': " ", "split": get_random_split()}])
                df = pd.concat([df, new_row], ignore_index=True)

    print(df)
    df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S")

    df = df.sort_values(by='datetime')
    df['datetime'] = df['datetime'].dt.strftime("%Y-%m-%d %H:%M:%S")
    print("Sorted df by date")

    df.to_csv(outcsv)

def remove_nolinks(incsv, outcsv):
    '''
    If the link field is empty, you cannot train with that datum

    Args:
        incsv (str): Path to the input metadata CSV file 
        outcsv (str): Path to the output CSV file after removing rows with no link field
    '''
    df = pd.read_csv(incsv)

    # Drop rows where 'link' is NaN or just whitespace
    df_cleaned = df[~(df['link'].isna() | (df['link'].str.strip() == ''))]
    df_cleaned.to_csv(outcsv)

def remove_broken_fits(incsv, outcsv, zips):
    '''
    Reads a CSV file containing filenames, attempts to extract and load FITS files 
    from a list of ZIP archives, and removes entries corresponding to broken or 
    unreadable FITS files.

    Args:
        incsv (str): Path to the input metadata CSV file 
        outcsv (str): Path to the output CSV file after removing broken entries
        zips (list of str): List of ZIP archive paths to search for the FITS files
    '''
    df = pd.read_csv(incsv)
    indices_to_remove = []

    for index, row in df.iterrows():
        try:
            for zip_path in zips:
                zip_ref = zipfile.ZipFile(zip_path, 'r')
                if row['filename'] in zip_ref.namelist():
                    fit_file = zip_ref.open(row['filename'])
                    fits_full_data = fits.open(fit_file)
                    spectrogram_arr = utils.load_fits_file(fits_full_data)
                    break
        except Exception as e:
            indices_to_remove.append(index)

    df.drop(indices_to_remove, inplace=True)
    df.to_csv(outcsv)

def mse_link_search(incsv, outcsv, zips, stations):
    '''
    Matches spectrogram data from a local dataset with external Callisto FITS files based on timestamps and station names,
    then calculates the Mean Squared Error (MSE) between corresponding images to find the best match.
    WARNING: this may take several hours to run. ~10sec per datum

    Args:
        incsv (str): Path to the input CSV file containing metadata, including filenames and timestamps.
        outcsv (str): Path to the output CSV file where updated metadata, including best-matching links and MSE values, will be saved.
        zips (list of str): List of paths to ZIP archives containing local FITS files.
        stations (list of str): List of station identifiers to filter Callisto files.
    
    '''
    Callisto_fit_url = "http://soleil80.cs.technik.fhnw.ch/solarradio/data/2002-20yy_Callisto/"

    df = pd.read_csv(incsv)

    current_date = None
    file_list = None
    i = 0
    start = time.perf_counter()
    for index, row in df.iterrows():
        utc = clock.est_to_utc(row['datetime'])
        new_date = clock.time_helper(utc)['date']
        new_time = clock.time_helper(utc)['time']

        #change the saved file list whenver the date changes
        if new_date != current_date:
            current_date = new_date
            current_datetime = utc
            url = Callisto_fit_url + clock.time_helper(current_datetime)['year'] + "/" + clock.time_helper(current_datetime)['month'] + "/" + clock.time_helper(current_datetime)['day'] + "/"
            file_list = web.list_files(url)

        new_time_fifteen = clock.round_to_nearest_15(utc)
        search_date = new_date.replace("-", "")
        search_time_exact = new_time.replace(":", "")
        search_time_fifteen = new_time_fifteen.replace(":", "")

        #if the date is wrong get rid of it
        matching_dates = [f for f in file_list if search_date in f]

        #only keep files with the exact time stamp or the time stamp rounded to the nearest quarter hour
        matching_dates_and_times = []
        for f in matching_dates:
            if search_time_exact in f or search_time_fifteen in f:
                matching_dates_and_times.append(f)

        #only keep if it is from one of the stations we listed
        matching_files = []
        for f in matching_dates_and_times:
            for s in stations:
                if s in f:
                    matching_files.append(f)
                    break

        #get the peach mountain data
        try:
            spectrogram_arr = []
            for zip_path in zips:
                zip_ref = zipfile.ZipFile(zip_path, 'r')
                if row['filename'] in zip_ref.namelist():
                    fit_file = zip_ref.open(row['filename'])
                    fits_full_data = fits.open(fit_file)
                    spectrogram_arr = utils.load_fits_file(fits_full_data)
                    break

            #find the callisto file that is most similar to the peach mountain data
            link = None
            best_mse = sys.float_info.max
            for u in matching_files:

                #get this callisto spectrogram
                response = requests.get(u, stream=True)

                spectrogram_arr_callisto = []
                with gzip.open(BytesIO(response.content), "rb") as gz_file:
                    decompressed_data = BytesIO(gz_file.read())
                    with fits.open(decompressed_data) as hdul:
                        spectrogram_arr_callisto = hdul[0].data  # Access primary data (numpy array)

                #resize callisto image to be same as peach mountain image
                if spectrogram_arr.shape != spectrogram_arr_callisto.shape:
                    spectrogram_arr_callisto = skimage.transform.resize(spectrogram_arr_callisto, spectrogram_arr.shape)

                #get mean squared error
                mse = np.mean((spectrogram_arr_callisto.astype("float32") - spectrogram_arr.astype("float32")) ** 2)

                #find best mse
                if mse < best_mse:
                    best_mse = mse
                    link = u
        
            #append to metadata
            if link:
                df.at[index, 'link'] = link
                df.at[index, 'initial_mse'] = best_mse
                print(link)
            else:
                print(f"File not found: {row['filename']}")

        except Exception as e:
            print(f"File not found: {row['filename']}")

        print(f"{i}/{len(df)} Complete. {(time.perf_counter() - start):.6f} seconds elasped")
        i += 1

    df.to_csv(outcsv)

def append_spectrogram_size(incsv, outcsv, zips):
    '''
    Extracts the dimensions (frequency and time) of spectrogram data from FITS files stored in ZIP archives
    and appends these dimensions as new columns in a CSV file.

    Args:
        incsv (str): Path to the input CSV file containing metadata, including filenames.
        outcsv (str): Path to the output CSV file where updated metadata, including spectrogram dimensions, will be saved.
        zips (list of str): List of paths to ZIP archives containing FITS files.
    '''
    df = pd.read_csv(incsv)

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
    

    df.to_csv(outcsv, index=False)

def sample_metadata(incsv, outcsv, factor):
    '''
    Creates a balanced dataset by keeping all rows where 'burst' is 1 and randomly sampling 
    a subset of rows where 'burst' is 0 at a specified ratio.

    Args:
        incsv (str): Path to the input CSV file.
        outcsv (str): Path to the output CSV file where the sampled dataset will be saved.
        factor (int): The ratio of non-burst to burst samples to include in the final dataset.
    '''
    df = pd.read_csv(incsv)
    df_burst = df[df["burst"] == 1]
    df_nonburst = df[df["burst"] == 0]
    df_nonburst = df_nonburst.sample(n=factor*len(df_burst), random_state=42)
    df_sample = pd.concat([df_burst, df_nonburst])
    df_sample.to_csv(outcsv)

def station_link_search(incsv, outcsv, station):
    '''
    Matches spectrogram data with corresponding Callisto files based on timestamps.
    This will find a Callisto file with the exact same timestamp as the Peach Mountain data (converted to UTC).
    This will select the first file found that includes the given station name.

    Args:
        incsv (str): Path to the input CSV file.
        outcsv (str): Path to save the updated CSV file.
        station (str): Identifier for the desired station's data.
    '''
    Callisto_fit_url = "http://soleil80.cs.technik.fhnw.ch/solarradio/data/2002-20yy_Callisto/"

    df = pd.read_csv(incsv)

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
        station_file = next((f for f in matching_dates_and_times if station in f), None)
        if station_file:
            df.at[index, 'link'] = station_file
            print(station_file)
        else:
            print(f"File not found: {station_file}")

    df.to_csv(outcsv)