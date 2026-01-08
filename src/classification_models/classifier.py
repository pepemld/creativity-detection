"""
    This script uses xlm-roberta, either from huggingface or a fine-tuned version stored locally, to classify sentences
"""

import argparse
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help="Path to input text file")
parser.add_argument('--output', type=str, required=True, help="Path to output file where results are stored")
parser.add_argument('--model-path', type=str, default='FacebookAI/roberta-large', help="Model path, both as HuggingFace ID or local checkpoint path")
args = parser.parse_args()

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# Read input file
input_df = pd.read_csv(args.input)

# Classify sentences
classifications = []
for id, row in input_df.iterrows():
    sentence = row['sentence']
    encoded_input = tokenizer(sentence, return_tensors='pt')
    output = model(**encoded_input)
    logits = output.logits
    probs = torch.softmax(logits, dim=-1)
    classification = int(torch.argmax(probs, dim=-1))
    confidence = float(torch.max(probs, dim=-1).values)

    classifications.append({'segment':row['segment'], 'sentence':sentence, 'classification':classification, 'confidence':confidence})

classifications_df = pd.DataFrame(classifications)

classifications_df.to_csv(args.output, index=False)