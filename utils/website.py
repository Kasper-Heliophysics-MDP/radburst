import requests
from bs4 import BeautifulSoup
import urllib.parse
import os
import re
import pandas as pd


def list_files(url, ext=None):
    '''
    Returns a list of all files of a given extension from a website

    Args:
        url (string): Website to search.
        ext (string): File extension to look for.
        
    Returns:
        list: contains all file urls
    '''

    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract all anchor tags
    links = soup.find_all("a")

    # Extract file URLs
    file_urls = []
    for link in links:
        href = link.get("href")
        if(ext):
            if href and (href.endswith(ext)):  # Adjust extensions as needed
                full_url = urllib.parse.urljoin(url, href)  # Handle relative URLs
                file_urls.append(full_url)
        else:
            if href:  # Adjust extensions as needed
                full_url = urllib.parse.urljoin(url, href)  # Handle relative URLs
                file_urls.append(full_url)

    return file_urls

def download_file(url, search_term):
    '''
    Downloads a file from url that contains search term in the filename.
    If there are multiple files with this search term it will download the first one.

    Args:
        url (string): Website to search.
        ext (string): Term to look for in file name.
        
    Returns:
        None
    '''
    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract all anchor tags
    links = soup.find_all("a")

    # Extract file URLs
    file_urls = []
    for link in links:
        href = link.get("href")
        if href and (href.endswith(".txt")):  # Adjust extensions as needed
            full_url = urllib.parse.urljoin(url, href)  # Handle relative URLs
            file_urls.append(full_url)

    for link in links:
        href = link.get("href")
        if href and search_term in href:
            full_url = urllib.parse.urljoin(url, href)  # Handle relative URLs
            
            # Extract filename from URL
            original_filename = os.path.basename(full_url)  # e.g., "2024_03_report.pdf"
            filename_stub, _ = os.path.splitext(original_filename)  # e.g., "2024_03_report"

            # Rename file to only contain the stub
            new_filename = filename_stub  # e.g., "2024_03_report" (no extension)

            # Download the file
            file_response = requests.get(full_url, stream=True)
            
            # Save the file
            with open(new_filename, "wb") as file:
                for chunk in file_response.iter_content(chunk_size=8192):
                    file.write(chunk)

def get_file_list(url, search_terms):
    '''
    Searches for a file from url that contains search term in the filename.
    Returns a reference to the content of this file without downloading.
    If there are multiple files with this search term it will download the first one.

    Args:
        url (string): Website to search.
        search_terms (list of strings): Terms to look for in file name.
        
    Returns:
        response:
            response.text: Returns the response content as a string (for text-based responses like HTML).
            response.content: Returns the raw binary content (useful for images, PDFs, etc.).
            response.json(): Parses the response as JSON (if applicable).
            response.status_code: HTTP status code (e.g., 200 for success, 404 for not found).
            response.headers: Returns response headers as a dictionary.
            response.url: Returns the final URL after redirections.
    '''
    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract all anchor tags
    links = soup.find_all("a")

    for link in links:
        href = link.get("href")
        if href and all(s in href for s in search_terms):
            full_url = urllib.parse.urljoin(url, href)  # Handle relative URLs
            response = requests.get(full_url, stream=True)
            return response
        
    # If a response isn't found function will print error info
    print(f"Error: file not found. Terms: {search_terms}")
    return None

def get_file(url, search_term):
    '''
    Searches for a file from url that contains search term in the filename.
    Returns a reference to the content of this file without downloading.
    If there are multiple files with this search term it will download the first one.

    Args:
        url (string): Website to search.
        search_terms (list of strings): Terms to look for in file name.
        
    Returns:
        response:
            response.text: Returns the response content as a string (for text-based responses like HTML).
            response.content: Returns the raw binary content (useful for images, PDFs, etc.).
            response.json(): Parses the response as JSON (if applicable).
            response.status_code: HTTP status code (e.g., 200 for success, 404 for not found).
            response.headers: Returns response headers as a dictionary.
            response.url: Returns the final URL after redirections.
    '''
    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract all anchor tags
    links = soup.find_all("a")

    for link in links:
        href = link.get("href")
        if href and search_term in href:
            full_url = urllib.parse.urljoin(url, href)  # Handle relative URLs
            response = requests.get(full_url, stream=True)
            return response
        
    # If a response isn't found function will print error info
    print(f"Error: file not found. Term: {search_term}")
    return None
        
def read_burst_list(response):
    '''
    Forms a dictionary linking burst times to stations that captured them

    Args:
        response: response from requests.get()

    Returns:
        pd.DataFrame: df containing info from the text file
    '''
    # Initialize df to store time ranges and their corresponding stations
    time_to_stations = pd.DataFrame(columns=['time', 'date', 'stations'])
    if response.status_code == 200:
        # Read the text file
        for line in response.iter_lines(decode_unicode=True):
            datestring = line[:8]
            match = re.match(r"(\d{8})\s+(\d{2}:\d{2}-\d{2}:\d{2})\s+.*?\s+(.+)", line.strip())
            if match:
                time_range = match.group(2)  # Extract time range
                stations = match.group(3).split(", ")  # Extract stations as a list
                row = pd.DataFrame([{'time': time_range, 'date': datestring, 'stations': stations}])
                time_to_stations = pd.concat([time_to_stations, row], ignore_index=True)

        return time_to_stations
        
    else:
        print(f"Error reading file: {response.url}")




    