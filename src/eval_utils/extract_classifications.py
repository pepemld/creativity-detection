"""
    This script extracts per-segment classifications, undoing the merges made in the alignment step.
"""

import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help="Path to input file")
parser.add_argument('--output', type=str, required=True, help="Path where classification is saved")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--percentile', type=int, help="Percentile threshold for high/low diversity.")
group.add_argument('--threshold', type=str, help="Fixed threshold for high/low diversity.")
args = parser.parse_args()


# Read CSV file
df = pd.read_csv(args.input)

if 'creativity_score' in df:
    creativity_score = 'creativity_score'
elif 'cosine_distance' in  df:
    creativity_score = 'cosine_distance'
else:
    raise Exception("No scores in the input file.")


# Find threshold value according to percentile
if args.percentile:
    threshold = np.percentile(df[creativity_score],100-args.percentile)
    print(f"Threshold for {args.percentile} percentile on file {args.input}: {threshold}")
elif args.threshold:
    threshold = float(args.threshold)
else:
    raise Exception("No threshold or percentile value has been set.")


# Perform classification
output = {'segment':[],'sentence':[],'classification':[]}
for i, row in df.iterrows():
    for segment, sentence in zip(row['segment'].split('~~~'),row['sentence'].split(' ~~~ ')):
        output['segment'].append(int(segment))
        output['sentence'].append(sentence)

        if row[creativity_score]>=threshold:
            output['classification'].append(1)
        else:
            output['classification'].append(0)

        
output_df = pd.DataFrame(output).sort_values('segment')
output_df.to_csv(args.output,index=False)