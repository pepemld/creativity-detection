"""
    This script calculates creativity as the cosine similarity between a text and its translation
"""

import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help="Path to input file. Must be an aligned CSV file containing a 'sentence' and a 'translation' column.")
parser.add_argument('--output', type=str, required=True, help="Path to output file.")
args = parser.parse_args()


data = pd.read_csv(args.input)

sentence_transformer = SentenceTransformer('distiluse-base-multilingual-cased-v2')


output_data = []
for id, row in data.iterrows():
    source_embeddings = sentence_transformer.encode(row['sentence'])
    transl_embeddings = sentence_transformer.encode(row['translation'])

    distance = cosine_distances([source_embeddings], [transl_embeddings])

    output_data.append({
        'segment':row['segment'],
        'sentence':row['sentence'],
        'translation':row['translation'],
        'cosine_distance':distance[0][0]
    })

output_df = pd.DataFrame(output_data)
sorted_df = output_df.sort_values('cosine_distance')
sorted_df.to_csv(args.output)