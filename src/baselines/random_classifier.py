"""
    This script classifies sentences randomly to create a baseline
"""

import argparse
import os
import pandas as pd
import random


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help="Path to input text file")
parser.add_argument('--output', type=str, required=True, help="Path to output file where results are stored")
args = parser.parse_args()


# Read input file
df = pd.read_csv(args.input)

# Generate random classifications
df['classification'] = [random.randint(0,1) for _ in range(len(df))]

# Save classification
df.to_csv(args.output, index=False)