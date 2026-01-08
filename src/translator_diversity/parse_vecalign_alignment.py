"""
    Parses vecalign alignment to pair corresponding sentences from two texts
"""

import argparse
import os
import re
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--alignment', required=True, help="Path to vecalign alignment file")
parser.add_argument('--source', required=True, help="Path to source text file")
parser.add_argument('--target', required=True, help="Path to target text file")
parser.add_argument('--output', required=True, help="Path where output CSV file will be saved")
args = parser.parse_args()


def parse_alignment_file():
    alignments = []

    with open(args.alignment,'r',encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # Parse format: [source_indices]:[target_indices]:confidence
            match = re.match(r'\[([^\]]*)\]:\[([^\]]*)\]:([0-9.]+)', line)
            if match:
                source_str, target_str, confidence_str = match.groups()
                if source_str == '' or target_str == '':
                   continue
                
                # Parse source indices
                source_indices = []
                if source_str.strip():
                    source_indices = [int(x.strip()) for x in source_str.split(',')]
                
                # Parse target indices
                target_indices = []
                if target_str.strip():
                    target_indices = [int(x.strip()) for x in target_str.split(',')]
                
                confidence = float(confidence_str)
                alignments.append((source_indices, target_indices, confidence))

    return alignments

def split_into_sentences(text):
    sentences = text.splitlines()
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def apply_alignment(source_sentences, target_sentences, alignments, translator):
    aligned_text = []

    initial_segment = 1
    for source_ids, target_ids, confidence in alignments:
        current_segment = initial_segment

        source_span = []
        for i in source_ids:
            if i < len(source_sentences):
                current_segment += 1
                source_span.append(source_sentences[i])
        
        target_span = []
        for i in target_ids:
            if i < len(target_sentences):
                target_span.append(target_sentences[i])
        
        segments = []
        for i in range(initial_segment,current_segment):
            segments.append(str(i))
        
        aligned_text.append({
            'segment': '~~~'.join(segments),
            'original': ' ~~~ '.join(source_span),
            translator: ' ~~~ '.join(target_span),
            'confidence': confidence
        })

        initial_segment = current_segment
    
    return aligned_text




if not os.path.isfile(args.alignment):
    raise FileExistsError(f"Alignment file not found: {args.alignment}")

if not os.path.isfile(args.source):
    raise FileExistsError(f"Source file not found: {args.source}")

if not os.path.isfile(args.target):
    raise FileExistsError(f"Target file not found: {args.target}")

output_dir = os.path.dirname(args.output)
if not os.path.isdir(output_dir):
    raise NotADirectoryError(f"Output folder does not exist: {args.output_dir}")


# Load alignments
alignments = parse_alignment_file()
print(f"Loaded {len(alignments)} alignments from {args.alignment}")

# Read files
with open(args.source, 'r', encoding='utf-8') as file:
    source_text = file.read()
source_sentences = split_into_sentences(source_text)

with open(args.target, 'r', encoding='utf-8') as file:
    target_text = file.read()
target_sentences = split_into_sentences(target_text)

translator = os.path.splitext(os.path.basename(args.target))[0]

# Align texts
aligned_text = apply_alignment(source_sentences, target_sentences, alignments, translator)

# Save to csv
pd.DataFrame(aligned_text).to_csv(args.output, index=False)
print(f"Alignment saved in {args.output}")