"""
    Format the cleaned text files such that they include no empty lines and one line per sentence.
"""

import argparse
import os
import re
import nltk

parser = argparse.ArgumentParser()
parser.add_argument('--bookdir', type=str, required=True, help="Path to the folder containing the book files")
args = parser.parse_args()

if not os.path.isdir(args.bookdir):
    raise NotADirectoryError(f"Book directory not found: {args.bookdir}")

input_dir = os.path.join(args.bookdir,"03_cleaned_text")
output_dir = os.path.join(args.bookdir,"04_formatted_text")

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

for file in os.listdir(input_dir):
    if not file.endswith('.txt'):
        continue

    with open(f"{input_dir}/{file}",'r') as in_file, open(f"{output_dir}/{file}",'w') as out_file:
        text = in_file.read()
        
        # remove artificial line breaks happening in the middle of sentences in some texts (Gutenberg .txt files)
        text = re.sub("\n(?!\n)",' ',text)
        # remove space at beginning of lines introduced as a side-effect of the previous step
        text = text.replace('\n ','\n')
        # remove all empty lines
        text = re.sub(r'^[ \t]*\n', '', text, flags=re.MULTILINE)
        # remove in-text footnotes
        text = re.sub(r'\[\d+\]', '', text)
        
        # split text into sentences, ready for alignment
        all_sentences = []
        for line in text.split('\n'):
            sentences = nltk.sent_tokenize(line)
            all_sentences.extend(sentences)

        text = '\n'.join(all_sentences)

        out_file.write(text)

print(f"Files in {args.bookdir} have been formatted")