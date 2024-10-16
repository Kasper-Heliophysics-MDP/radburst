- Explain datasets here (collected data, online ecallisto, etc) including where/how to download them
- Include folder here for sample data w/ labels to demonstrate or test functions (preprocessing, detection, etc)

# Data Download and Processing

## e-Callisto Data

### 1. Data Source

- **Source**: e-Callisto (extended Compound Astronomical Low-cost Low-frequency Instrument for Spectroscopy and Transportable Observatory) network
- **URL**: http://soleil80.cs.technik.fhnw.ch/solarradio/data/2002-20yy_Callisto/
- **Content**: Solar radio spectrograms from various observatories worldwide
- **Structure**: Organized by year/month/day

### 2. Downloaded Data

- **Location**: Local 'data/' directory
- **Structure**: Mirrors the online repository (year/month/day)
- **File Type**: .gz files (likely compressed FITS files)

### 3. Processed Data

- **Process**: Files are filtered based on a maximum frequency threshold
- **Output**:
  - 'all_files.txt': Lists all downloaded files
  - 'kept_files.txt': Lists files kept after filtering

## Hugging Face Datasets

Australia ASAA data.
