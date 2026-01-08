"""
    This script joins all the classification files in .csv format in a specified directory into a singular dataset.
"""

import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True, help="Name of folder where the separate classifications are stored")
parser.add_argument('--output', type=str, required=True, help="Name of the file where the resulting dataset will be stored")
args = parser.parse_args()

if not os.path.isdir(args.input_dir):
    raise NotADirectoryError(f"Input directory not found: {args.input_dir}")

segment = 1
dataset = []
for filename in os.listdir(args.input_dir):
    if not filename.endswith('.csv'):
        continue

    df = pd.read_csv(f"{args.input_dir}/{filename}")
    
    for i, row in df.iterrows():
        dataset.append({'segment':segment, 'sentence':row['sentence'], 'classification':row['classification']})


dataset_df = pd.DataFrame(dataset)
dataset_df.to_csv(args.output, index=False)