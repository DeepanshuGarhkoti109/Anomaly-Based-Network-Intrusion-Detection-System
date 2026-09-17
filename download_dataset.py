#!/usr/bin/env python3
"""
Script to download UNSW-NB15 dataset from GitHub.
"""

import os
import requests
from tqdm import tqdm

def download_file(url, output_path):
    """Download a file with progress bar."""
    print(f"Downloading {url} to {output_path}")
    
    # Stream the download to handle large files
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get total file size
    total_size = int(response.headers.get('content-length', 0))
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Download with progress bar
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(output_path)) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Downloaded {output_path} ({os.path.getsize(output_path) / (1024*1024):.2f} MB)")

def main():
    """Main function to download dataset."""
    
    # GitHub raw URLs for the dataset
    base_url = "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/"
    
    files = [
        ("UNSW_NB15_training-set.csv", "data/raw/UNSW_NB15_training-set.csv"),
        ("UNSW_NB15_testing-set.csv", "data/raw/UNSW_NB15_testing-set.csv")
    ]
    
    print("Downloading UNSW-NB15 dataset...")
    
    for filename, output_path in files:
        url = base_url + filename
        try:
            download_file(url, output_path)
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            print("You may need to download the dataset manually from:")
            print("https://research.unsw.edu.au/projects/unsw-nb15-dataset")
            print(f"Place the file at: {output_path}")
    
    print("\nVerifying downloaded files...")
    
    for _, output_path in files:
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024*1024)
            print(f"✓ {output_path}: {size_mb:.2f} MB")
        else:
            print(f"✗ {output_path}: File not found")
    
    print("\nDone!")

if __name__ == "__main__":
    main()