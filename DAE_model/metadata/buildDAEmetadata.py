from metautils import *

# Modules for creating and editing metadata files used in DAE training
# Note than this module format may be better suited for Jupyter Notebook
# However, the author of this code has a personal vendetta against Jupyter Notebook
# So, the if statements are set up to achieve a similar effect

zips = ['data/FITfiles-20250205T173408Z-001.zip', 'data/FITfiles-20250205T173408Z-002.zip', 'data/FITfiles-20250205T173408Z-003.zip']
metadata_path = 'DAE_model/metadata/DAEmetadata_Alaska_sample.csv'

#make the csv
if(0):
    labels_csv = 'data/metadata/classification_labels_raw.csv'
    init_metadata(metadata_path, zips, labels_csv)

#find calliso datalink by selecting the first file with a matching time stamp from alaska
if(0):
    station_link_search(metadata_path, metadata_path, "ALASKA")

#find callisto datalink with by selecting data from around the same time at any of the listed stations that has the lowest initial mse
if(0):
    stations = ["ALASKA-COHOE", "ALASKA-HAARP", "GREENLAND", "MEXICO-LANCE", "BIR", "MEXART", "GERMANY-DLR", "SWISS-HEITERSWIL", "USA-BOSTON"]
    mse_link_search(metadata_path, metadata_path, zips, stations)

#calc peach mountain data size
if(0):
    append_spectrogram_size(metadata_path, metadata_path, zips)

#get a random sample
if(0):
    sample_path = 'DAE_model/metadata/DAEmetadata_Alaska_sample.csv'
    sample_metadata(metadata_path, sample_path, 5)

#filter out broken peach mountain files
if(0):
    remove_broken_fits(metadata_path, metadata_path, zips)

#filter out data pieces with no callisto link
if(1):
    remove_nolinks(metadata_path, metadata_path)