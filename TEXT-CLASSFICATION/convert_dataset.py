import os
import csv
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Path to the main folder containing the subfolders
main_folder = 'main_folder'

# Initialize an empty list to hold file names and BIRADS scores
data = []

# Function to normalize BIRADS score
def normalize_birads_score(score):
    # Remove non-digit characters except for the digit itself
    match = re.search(r'(\d)', score)
    return match.group(1) if match else None

# Function to extract BIRADS score from text
def extract_birads_score(text):
    lines = text.split('\n')
    for line in lines:
        if 'BIRADS' in line or 'SONUÇ' in line:
            normalized_score = normalize_birads_score(line)
            if normalized_score:
                return normalized_score
    return None

# Function to remove the BIRADS score section from text
def remove_birads_score(text):
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Check if the line contains the BIRADS score indication
        if not ('BIRADS' in line or 'SONUÇ' in line):
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

# Traverse the main folder and its subfolders
subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir()]

for subfolder in subfolders:
    for root, dirs, files in os.walk(subfolder):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    birads_score = extract_birads_score(content)
                    if birads_score:
                        data.append([file, birads_score])
                    # Remove BIRADS score section
                    cleaned_content = remove_birads_score(content)
                # Write the cleaned content back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)

# Define the path to save the CSV file
csv_file_path = 'output2.csv'

# Save the data to a CSV file
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['File Name', 'BIRADS Score'])
    writer.writerows(data)

print(f"CSV file has been created successfully at {csv_file_path}")
print("BIRADS score sections have been removed from all text files.")

# Read the existing CSV file
csv_data = []
with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)
    csv_data = list(reader)

# Add a new header for the Content column
headers.append('Content')

# Dictionary to map file names to their contents
file_content_map = {}

# Traverse the main folder and its subfolders again to get content
for subfolder in subfolders:
    for root, dirs, files in os.walk(subfolder):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                file_content_map[file] = file_content

# Update the CSV data with file contents
for row in csv_data:
    file_name = row[0]
    if file_name in file_content_map:
        row.append(file_content_map[file_name])
    else:
        row.append('')

# Write the updated data back to a new CSV file
new_csv_file_path = 'database.csv'

with open(new_csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)
    writer.writerows(csv_data)

print(f"CSV file has been updated successfully at {new_csv_file_path}")

# Load the new CSV into a DataFrame and shuffle it
df = pd.read_csv(new_csv_file_path)
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to a CSV file
shuffled_df.to_csv('main_dataframe.csv', index=False)

print("DataFrame has been shuffled and saved as main_dataframe.csv")
