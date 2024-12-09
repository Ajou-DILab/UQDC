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
