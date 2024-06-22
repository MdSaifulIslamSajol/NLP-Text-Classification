#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 23:27:12 2024

@author: saiful
"""
# https://github.com/SALT-NLP/implicit-hate
#%%
import os
import pandas as pd

# Define the base directory and paths to relevant folders and files
base_dir = "/home/saiful/bangla fault news/hate speech detection/hate-speech-dataset/hate-speech-dataset-master/"  # Update this path to your dataset directory
all_files_dir = os.path.join(base_dir, 'all_files')
annotations_file = os.path.join(base_dir, 'annotations_metadata.csv')

# Load the annotations metadata
annotations_df = pd.read_csv(annotations_file)

# Initialize an empty list to hold the combined data
combined_data = []

# Iterate over each file in the all_files directory
for file_name in os.listdir(all_files_dir):
    if file_name.endswith('.txt'):
        file_path = os.path.join(all_files_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            combined_data.append([file_name.replace('.txt', ''), content])

# Create a DataFrame from the combined data
combined_df = pd.DataFrame(combined_data, columns=['file_id', 'text'])

# Merge with annotations metadata to get the labels and text together
final_df = pd.merge(annotations_df, combined_df, on='file_id')

# Display the final dataframe
print(final_df.head())

# Save the final dataframe to a CSV file if needed
final_df.to_csv(os.path.join(base_dir, 'combined_data_with_text.csv'), index=False)
