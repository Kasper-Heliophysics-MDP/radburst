import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import zipfile
import requests
import shutil
import stat
import matplotlib.pyplot as plt
import sys
import random
import gzip
from astropy.io import fits
from io import BytesIO
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset import Dataset
import utils.utils as utils
import utils.time_utils as clock
import utils.website as web

pull_from_month = "04"
station = "ALASKA"
Callisto_fit_url = "http://soleil80.cs.technik.fhnw.ch/solarradio/data/2002-20yy_Callisto/"
data_path = ['data/FITfiles-20250205T173408Z-001.zip', 'data/FITfiles-20250205T173408Z-002.zip', 'data/FITfiles-20250205T173408Z-003.zip']
labels_path = 'data/metadata/filtered-labels-20240309-20240701.csv'

# Create a Dataset object which loads all data from the given path (defined in dataset.py)
data = Dataset(data_dir= data_path,
              labels= labels_path,
              zip=True)
data = data.only_bursts()

for i in range(0, 3):
    dat_dict = random.choice(data)
    while(web.time_helper(dat_dict['datetime'])['month'] != pull_from_month):
        print(web.time_helper(dat_dict['datetime'])['month'])
        dat_dict = random.choice(data)

    file_path = os.path.join('FITfiles/', dat_dict['path'])
    data_collect = 'data_to_send'
    spectrogram_arr = None
    file_found = False
    output_jpg = None
    os.makedirs(data_collect, exist_ok=True)
    for zipped_file in data_path:
        zip_ref = zipfile.ZipFile(zipped_file, 'r')
        fit_file_name = file_path
        file_found = False
        if fit_file_name in zip_ref.namelist():
            file_found = True
            fit_file = zip_ref.open(fit_file_name)
            fits_full_data = fits.open(fit_file)
            spectrogram_arr = utils.load_fits_file(fits_full_data)
            plt.imshow(spectrogram_arr.astype(float), aspect='auto')  
            plt.title(fit_file_name)

            # Save the plot as a .jpg file
            output_jpg = os.path.join(os.path.join(data_collect, "unclassified_files"), os.path.splitext(os.path.basename(fit_file_name))[0] + '.jpg')
            plt.savefig(output_jpg, format='jpg', dpi=300)  # Save with 300 DPI for better quality
            plt.close() 
            print(output_jpg)
            break
    
    if(file_found):
        utc = clock.est_to_utc(dat_dict['datetime'])
        url = Callisto_fit_url + clock.time_helper(utc)['year'] + "/" + clock.time_helper(utc)['month'] + "/" + clock.time_helper(utc)['day'] + "/"
        print(url)
        print([station, clock.time_helper(utc)['time'].replace(":", "")])
        response = web.get_file_list(url, [station, clock.time_helper(utc)['time'].replace(":", "")])
        print(response.status_code)  # Should be 200 if successful
        print(response.headers.get("Content-Type"))  # Should be 'application/fits' or similar
        with gzip.open(BytesIO(response.content), "rb") as gz_file:
            decompressed_data = BytesIO(gz_file.read())
            with fits.open(decompressed_data) as hdul:
                hdul.info()  # Display FITS file structure
                spectrogram_arr = hdul[0].data  # Access primary data (numpy array)
                plt.imshow(spectrogram_arr.astype(float), aspect='auto')  
                plt.title(station + "_" + dat_dict['datetime'])

                # Save the plot as a .jpg file
                output_jpg = os.path.join(os.path.join(data_collect, "callisto_files"), station + "_" + dat_dict['datetime'].replace(" ", "-").replace(":", "-") + '.jpg')
                plt.savefig(output_jpg, format='jpg', dpi=300)  # Save with 300 DPI for better quality
                plt.close() 

csv_url = "http://soleil80.cs.technik.fhnw.ch/solarradio/data/BurstLists/2010-yyyy_Monstein/2024/"
response = web.get_file(csv_url, pull_from_month)
csv_path = os.path.join(data_collect, pull_from_month + "_burst_list.txt")
with open(csv_path, "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):  # Download in chunks
        file.write(chunk)
   
# Create a ZIP archive
zip_folder_name = 'zipped_data'
shutil.make_archive(zip_folder_name, "zip", data_collect)

zip_folder_name = zip_folder_name + ".zip"
if(0):
    # Attach the ZIP file

    # Email credentials
    sender_email = "callen.fields@gmail.com"
    password = "qtct xrvk ggut weli"  # Use an app password instead of your real password

    # Create the email

    uniqnames = ['aashim', 'fcallen']
    # Send the email
    for name in uniqnames:
        try:
            msg = MIMEMultipart()
            msg["From"] = sender_email
            receiver_email = name + "@umich.edu"
            msg["To"] = receiver_email
            msg["Subject"] = "Hello from Python!"

            body = "Hi Aashi\n\nThis is what I've been working on. tldr: we have 6 months of LWA data we could go through and classify to help with DAE training. This presents a way to automate this in an organized manner.\n\n"
            bod_ds = "There's about 2000 unclassified files in the google drive. We could apply a minimum frequency value threshold to get this number down a little. Also we probably don't need to do all of them. Really we just need to find more bursts so that our training dataset is more robust. "
            body_2 = "We could send this zip file out along with a google form. We would ask people to classify the unclassified files in this attached zip folder. Then they would fill out the form with their predictions. "
            body_3 = "It's all automated in python so we could easily adjust who we send an email to, how much data to send, and what files to send. Also, we can very easily keep track of every made prediction using results from the google form."
            body_4 = " Google Forms allows you to export results to a CSV so we could easily parse that in python to automate labeling as well.\n\n"
            body_5 = "But yeah, I won't be there today for my midterm so keep me updated. After I send this email, I will not be thinking about Helio until March 9th at the earliest.\n\n"
            body_6 = "Thank you,\nCallen"
            msg.attach(MIMEText(body + body_2 + body_3 + body_4 + body_5 + body_6, "plain"))

            with open(zip_folder_name, "rb") as attachment:
                part = MIMEApplication(attachment.read(), Name=os.path.basename(zip_folder_name))
                part["Content-Disposition"] = f'attachment; filename="{os.path.basename(zip_folder_name)}"'
                msg.attach(part)

            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()  # Secure the connection
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, msg.as_string())
                print("Email with ZIP file sent successfully!")
        except Exception as e:
            print(f"Error: {e}")

    def remove_readonly(func, path, _):
        os.chmod(path, stat.S_IWRITE)  # Remove read-only
        func(path)

    shutil.rmtree(data_collect, onerror=remove_readonly)
    os.remove(zip_folder_name)