import sys
import os
import zipfile
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.website as web
import utils.time_utils as clock

def get_start(entry):
    return entry['date'] + "_" + entry['time'][0:5]

def get_end(entry):
    return entry['date'] + "_" + entry['time'][6:11]
if(0):
    url = "https://soleil.i4ds.ch/solarradio/data/BurstLists/2010-yyyy_Monstein/2024/"
    search_months = ['03', '04', '05', '06', '07', '08', '09', '10', '11']
    burst_dates = []
    for month in search_months:
        # Load file and make dictionary
        response = web.get_file(url, "2024_" + month)
        callisto_data = web.read_burst_list(response)

        for index, entry in callisto_data.iterrows():
            start_time = clock.round_down_to_15(clock.utc_to_est(get_start(entry)))
            print("start")
            print(get_start(entry))
            print(clock.utc_to_est(get_start(entry)))
            print(start_time)
            #print(get_end(entry))
            end_time = clock.round_up_to_15(clock.utc_to_est(get_end(entry)))
            print("end")
            print(get_end(entry))
            print(clock.utc_to_est(get_end(entry)))
            print(end_time)
            
            #Typo in callisto data
            if get_start(entry) == "20240731_03:09":
                continue

            while start_time != end_time:
                burst_dates.append(start_time)
                start_time = clock.add_15_minutes(start_time)

i = 0
if(1):
    zips = ['data/FITfiles-20250205T173408Z-001.zip', 'data/FITfiles-20250205T173408Z-002.zip', 'data/FITfiles-20250205T173408Z-003.zip']
    output_csv = "data/metadata/potential-bursts.csv"
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for zip_path in zips:
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                # Iterate through files in the zip
                for file_name in zipf.namelist():
                    i = i + 1
                    # for string in burst_dates[:]:  # Iterate over a copy to modify the list
                    #     if string in file_name:
                    #         csv_writer.writerow([file_name])  # Write to CSV
                    #         burst_dates.remove(string)  # Remove matched string
                    #         break  # Stop checking this file once a match is found
print(i)
