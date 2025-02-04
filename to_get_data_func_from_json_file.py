import os
import json
import csv

# Define the base directory
base_dir = "/home/iv285/rds/rds-cam-clin-qmin-nCu8GHKxWRw/data/BIDS/CPFT_BIDS/derivatives/micapipe_v0.2.0"
print("Script started.")
print(f"Base directory: {base_dir}")

# Load the subject list
subject_list_file = "Alzheimers_QMIN_MC_WIBIC_NO_first_session.txt"
with open(subject_list_file, "r") as file:
    subject_list = [line.strip() for line in file]
print(f"Loaded subject list with {len(subject_list)} entries.")
print("First few subjects:", subject_list[:5])

# Define the output
output_dir = "/home/iv285/rds/rds-cam-clin-qmin-nCu8GHKxWRw/data/Iryna_analysis"
output_file = os.path.join(output_dir, "ImageType_func.csv")
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Open CSV
with open(output_file, "a", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    print("CSV file opened for writing.")

    # Process each folder
    for subj_folder in os.listdir(base_dir):
        print(f"Checking folder: {subj_folder}")
        if subj_folder.startswith("sub-") and subj_folder[4:] in subject_list:
            print(f"Processing subject folder: {subj_folder}")
            subj_path = os.path.join(base_dir, subj_folder)
            for ses_folder in os.listdir(subj_path):
                if ses_folder.startswith("ses"):
                    print(f"Processing session folder: {ses_folder}")
                    func_folder = os.path.join(subj_path, ses_folder, "func")
                    if os.path.isdir(func_folder):
                        for json_file in os.listdir(func_folder):
                            if json_file.endswith("bold.json"):
                                print(f"Processing JSON file: {json_file}")
                                with open(os.path.join(func_folder, json_file)) as f:
                                    json_data = json.load(f)
                                    series_description = json_data.get("SeriesDescription", "N/A")
                                    image_type = json_data.get("ImageType", "N/A")
                                    print(f"SeriesDescription: {series_description}, ImageType: {image_type}")
                                    csvwriter.writerow([base_dir, subj_folder, ses_folder, json_file, series_description, image_type])
                                    print("Data written to CSV.")
