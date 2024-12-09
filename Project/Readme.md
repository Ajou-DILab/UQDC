## Downloading the Sample Data
To download the sample data programmatically, use the following Python code:

```python
import requests
import pandas as pd

def download_data_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    session = requests.Session()
    response = session.get(url, stream=True)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            url = f"{url}&confirm={value}"
            response = session.get(url, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

# Example usage
file_id = "1ASGn3Ob8VUPgNJ8pfllwBchUpEK0gwyR"  # Replace with your file ID
destination = "data/sample.data"  # Path to save the file

# Download and load the DataFrame
download_data_from_google_drive(file_id, destination)
df = pd.read_csv(destination)
print(df.head())

## Downloading the Model
To download the trained model programmatically, use the following Python code:

```python
import requests

def download_model_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    session = requests.Session()
    response = session.get(url, stream=True)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            url = f"{url}&confirm={value}"
            response = session.get(url, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

# Example usage
file_id = "1abcDEFghiJKLmNO"  # Replace with your file ID
destination = "models/Training.pt"
download_model_from_google_drive(file_id, destination)

