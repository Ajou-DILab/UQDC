import requests
import pandas as pd

def download_data_from_google_drive(file_id, destination):
    """
    Download a file from Google Drive using its file ID.
    """
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    session = requests.Session()
    response = session.get(url, stream=True)

    # Handle potential warnings (e.g., large file confirmation)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            url = f"{url}&confirm={value}"
            response = session.get(url, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:  # Filter out keep-alive new chunks
                f.write(chunk)

# Example usage
file_id = "1V7mmgLRuT-M9fVcG2KKtfNV1uqlCfgZi"  # Replace with your file ID
destination = "data/sample.csv"  # Path to save the file

# Create the directory if it doesn't exist
import os
os.makedirs("data", exist_ok=True)

# Download the file
download_data_from_google_drive(file_id, destination)

# Load the DataFrame
df = pd.read_csv(destination)  # Assuming it's a CSV file
print(df.head())  # Display the first few rows
