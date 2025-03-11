# Data Download and Processing

**To read in data, download FITfiles from the google drive as a zip file. This will download multiple zip files. Place them all in this data folder**

    metadata/                               Folder of csv files for metadata.  

        classification_labels_raw.csv           Someone made manual classifications of sun data. These were png images of spectograms. Link to classifications: https://docs.google.com/document/d/1YfonaM4mR5wb6eVOeXuNQUpLTaiJpuKH_D6yrXy9aA4/edit  

        filtered-labels-20240309-20240701.csv   This csv contains rows for every fit file in the google drive from the range of dates (which believe is unknown) where classifications were made. **This csv is what you should use to read in data**

        full-labels-20240120-20241009.csv       This csv contains rows for every fit file in the google drive as of 10/18/24 with classifications listed.  

        burst_matching_stations                 For each burst datum we have, list eCallisto stations that claim to have a burst with duration at the same time as we our datum (+- 10mins)

        nonburst_but_matching_stations.csv      For each nonburst datum we have, list eCallisto stations with bursts. There is nothing here that is good

        eCallisto_station_matches.csv           Summary of each eCallisto station and how many matching bursts/nonmatching nonbursts they have

        potential-bursts.csv                    Every datum we have where at least one eCallisto station claims to have seen a burst at the time where that datum was recorded 

        DAEmetadata.csv                         Metadata to use for DAE training (Note: idk what unnamed is and don't know how to get rid of it)

    sample_data/                            Just examples to give you a picture of what we're working with.  
  
    create_labels.ipynb                     Builds the full labels and filtered labels csv. Just reads files from the drive and assigns them their classification as given in classification_labels_raw.csv.  

    buildDAEmetadata.py                     Takes every datum we have, builds a csv with its date, time, and burst classification, thenfinds a datum from any ALASKA station recorded at the same time and adds that to the csv

    potential_bursts.py                     For any eCallisto burst list (specified by search_months) takes every burst duration, expands the duration to be a multiple of 15 minutes, then adds any datum we have recorded inside that duration to potential-bursts.csv

    send_email.py                           Script for crowdsourcing data collection. It packs our data, Callisto data, and Callisto burst lists into a zip file and sending it over email
                                          
    query.py                                This searches our data against the eCallisto data to find matching bursts (we say burst, Callisto says burst) and nonmatching nonbursts (we say nonburst, Callisto says burst)


