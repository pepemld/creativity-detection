"""
    Merge the aligments made separately using vecalign and parse them into readable text.
"""

import argparse
import os
import re
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--alignments', type=str, required=True, help="Name of folder where the alignment files are found")
parser.add_argument('--book', type=str, required=True, help="Name of folder where the original texts are found")
parser.add_argument('--output', type=str, required=True, help="Name of output file")
args = parser.parse_args()


translators = []
alignments = {}
num_original_lines = 0
merged_sources = []
merged_translations = {}
merged_alignment = []
output_alignment = {}


def load_alignments():
    global num_original_lines

    if not os.path.isdir(args.alignments):
        raise NotADirectoryError(f"Alignment directory not found: {args.alignments}")

    for filename in os.listdir(args.alignments):
        if not filename.endswith('_raw.txt'):
            continue
        
        translator = filename.split('_raw.txt')[0]
        translators.append(translator)
        file_path = os.path.join(args.alignments, filename)

        with open(file_path,'r',encoding="utf-8") as file:
            lines = file.read().split('\n')
            lines.pop()
            num_original_lines = int(re.sub(r'^\[|\]$', '', lines[-1].split(':')[0]).split(', ')[-1])

        # parse columns
        alignments[translator] = {}
        for line in lines:
            og, tr, _ = line.split(':')
            if og == '[]' or tr == '[]':
                continue
            og = [int(x) for x in re.sub(r'^\[|\]$', '', og).split(', ')]
            tr = [int(x) for x in re.sub(r'^\[|\]$', '', tr).split(', ')]
            
            alignments[translator][og[0]] = tr


def merge_sources():
    # Find the minimum id of each span after all merges
    min_spans = range(0,num_original_lines+1)
    for translator, alignment in alignments.items():
        transl_min_spans = [i for i in alignment.keys()]
        min_spans = [x for x in min_spans if x in transl_min_spans]

    # Find spans to be merged
    for i in min_spans:
        current_span = [i]
        j=1
        while i+j not in min_spans and i+j<min_spans[-1]:
            current_span.append(i+j)
            j+=1
        merged_sources.append(current_span)

    
def merge_all_alignments():
    # Initialize merged_translations
    for translator in translators:
        merged_translations[translator] = []

    for span in merged_sources:
        merged_span = {'original':span}
        for translator, alignment in alignments.items():
            current_alignment =  []
            for i in span:
                if i in alignment:
                    current_alignment.extend(alignment[i])
            merged_span[translator] = current_alignment
        
        merged_alignment.append(merged_span)


def parse_alignments():
    # Populate empty dataframe to fill as we go
    data = np.empty((len(merged_alignment),len(translators)+2))
    output = pd.DataFrame(data, dtype=str, columns=['segment','original']+translators)
    
    # Fill in columns
    for translator in ['original']+translators:
        with open(f"{args.book}/{translator}.txt",'r',encoding='utf-8') as file:
            lines = file.read().split('\n')
            for i,span in enumerate(merged_alignment):
                current_span = []
                current_line = []
                
                for s in span[translator]:
                    current_span.append(str(s+1))
                    current_line.append(lines[s])

                if translator=='original':
                    output.loc[i,'segment'] = '~~~'.join(current_span)
                output.loc[i,translator] = ' ~~~ '.join(current_line)

    output.to_csv(args.output, index=False)




load_alignments()

merge_sources()

merge_all_alignments()

parse_alignments()

print(f"Successfully merged alignments and saved them to {args.output}")