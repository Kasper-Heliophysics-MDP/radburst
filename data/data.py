import zipfile

zip_path = "data/FITfiles-20250205T173408Z-001.zip"  # Path to your zip file

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    file_list = zip_ref.namelist()  # List all files in the zip
    print(file_list)