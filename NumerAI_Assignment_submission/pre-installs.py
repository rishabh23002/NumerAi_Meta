import os
import yaml
import gdown
import zipfile
import argparse
from numerapi import NumerAPI


Parse = argparse.ArgumentParser()
Parse.add_argument("--config", type=str, default = 'Assignment1_config.yaml')
Arguments = Parse.parse_args()

# Read Configuration file
with open(Arguments.config, "r") as F: configs = yaml.safe_load(F); F.close()

napi = NumerAPI()

# for file_name in configs["Dataset"][configs["Dataset"]["name"]]["files_to_download"]:
#     napi.download_dataset(filename = f"{file_name}",
#                           dest_path = f"data/{file_name}")


# https://drive.google.com/file/d/1VgjfPxh67KflhMxta_to6sBC20FJAOHv/view?usp=sharing
# Download Saved_models
drive_link = 'https://drive.google.com/uc?id=1VgjfPxh67KflhMxta_to6sBC20FJAOHv' 
zip_file_path = 'saved_models.zip'
gdown.download(drive_link, zip_file_path, quiet=False)

# Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall()

os.remove(zip_file_path)