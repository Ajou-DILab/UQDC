import requests

def download_model_from_google_drive(file_id, destination):
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
file_id = "1ASGn3Ob8VUPgNJ8pfllwBchUpEK0gwyR"  # Replace with your file ID
destination = "models/AL_model.pt"
download_model_from_google_drive(file_id, destination)
