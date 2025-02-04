import os
import shutil

subject_list_file = "/home/iv285/rds/rds-cam-clin-qmin-nCu8GHKxWRw/data/Iryna_analysis/Alzheimers_QMIN_MC_WIBIC_NO_first_session.txt"  # Update with your file path
source_directory = "/home/iv285/rds/rds-cam-clin-qmin-nCu8GHKxWRw/data/BIDS/CUH_BIDS/derivatives/micapipe_v0.2.0"
destination_directory = "/home/iv285/rds/rds-cam-clin-qmin-nCu8GHKxWRw/data/Iryna_analysis/Alzheimer_SUB_prep"  # Update with your destination path

# Create destination directory if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)

# Read subject IDs from the file
with open(subject_list_file, "r") as file:
    subject_ids = [line.strip() for line in file if line.strip()]

# Iterate through the source directory
for subdir in os.listdir(source_directory):
    if any(subid in subdir for subid in subject_ids):
        source_path = os.path.join(source_directory, subdir)
        dest_path = os.path.join(destination_directory, subdir)
        if os.path.isdir(source_path):  # Ensure it's a directory
            print(f"Copying {subdir} to {destination_directory}...")
            shutil.copytree(source_path, dest_path)

print("Copying complete!")