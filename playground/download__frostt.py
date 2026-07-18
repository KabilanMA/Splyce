import requests
import os

# Base URL for the FROSTT S3 bucket
BASE_URL = "https://s3.us-east-2.amazonaws.com/frostt/frostt_data"

# List of datasets based on the FROSTT repository structure
# Structure: (folder_name, file_name)
datasets = [
    ("amazon", "amazon-reviews.tns.gz"),
    ("chicago", "chicago-crime.tns.gz"),
    ("darpa", "darpa.tns.gz"),
    ("delicious", "delicious.tns.gz"),
    ("enron", "enron-emails.tns.gz"),
    ("freebase-music", "freebase-music.tns.gz"),
    ("flickr", "flickr.tns.gz"),
    ("lanl", "lanl-network-traffic.tns.gz"),
    ("lbnl", "lbnl-network.tns.gz"),
    ("nell", "nell-1.tns.gz"),
    ("nell", "nell-2.tns.gz"),
    ("nips", "nips-publications.tns.gz"),
    ("patents", "patents.tns.gz"),
    ("reddit", "reddit-2015.tns.gz"),
    ("uber", "uber-pickups.tns.gz"),
    ("vast-2015-mc1", "vast-2015-mc1.tns.gz"),
]

def download_datasets():
    if not os.path.exists("frostt_data"):
        os.makedirs("frostt_data")

    for folder, filename in datasets:
        url = f"{BASE_URL}/{folder}/{filename}"
        local_path = os.path.join("frostt_data", filename)
        
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Finished downloading {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    download_datasets()