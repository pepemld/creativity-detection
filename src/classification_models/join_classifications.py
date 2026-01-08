"""
    This script joins all the .txt files in a specified directory into a singular corpus.
    It also introduces a 'segment' column with sentence IDs to match the expected format of the annotation scripts.
"""

import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True, help="Name of folder where the original texts are stored")
parser.add_argument('--output', type=str, required=True, help="Name of the file where the resulting corpus will be stored")
args = parser.parse_args()

if not os.path.isdir(args.input_dir):
    raise NotADirectoryError(f"Input directory not found: {args.input_dir}")

segment = 1
corpus = []
for filename in os.listdir(args.input_dir):
    if not filename.endswith('.csv'):
        continue

    if '2br02b' in filename:
        continue

    input_data = pd.read_csv(f"{args.input_dir}/{filename}")
    for id, row in input_data.iterrows():
        corpus.append({'sentence':row['sentence'], 'classification':row['classification']})

corpus_df = pd.DataFrame(corpus)
corpus_df.to_csv(args.output, index=False)