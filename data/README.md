# Data Download and Processing

**To read in data, download FITfiles from the google drive as a zip file. This will download multiple zip files. Place them all in this data folder**

metadata/                               Folder of csv files for metadata.  

    classification_labels_raw.csv           Someone made manual classifications of sun data. These were png images of spectograms. 
                                            Link to classifications: https://docs.google.com/document/d/1YfonaM4mR5wb6eVOeXuNQUpLTaiJpuKH_D6yrXy9aA4/edit  

    filtered-labels-20240309-20240701.csv   This csv contains rows for every fit file in the google drive from the range of dates 
                                            (which believe is unknown) where classifications were made. __This csv is what you 
                                            should use to read in data.__ 

    full-labels-20240120-20241009.csv       This csv contains rows for every fit file in the google drive as of 
                                            10/18/24 with classifications listed.  

sample_data/                            Just examples to give you a picture of what we're working with.  
  
create_labels.ipynb                     Builds the full labels and filtered labels csv. Just reads files from the drive and 
                                        assigns them their classification as given in classification_labels_raw.csv.  
  
 

