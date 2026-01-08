"""
    Convert original epub files to .txt format
"""

import argparse
import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup as bs

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help="Path to the file in epub format")
parser.add_argument('--output', type=str, required=True, help="Path where txt file will be saved")
args = parser.parse_args()   

if not os.path.isfile(args.input):
    raise FileExistsError(f"Input file not found: {args.input}")

if not args.input.endswith('.epub'):
    raise FileExistsError(f"Input file must be in epub format")

output_dir = os.path.dirname(args.output)
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# Read the epub file
book = epub.read_epub(args.input)

# Extract text from all document items
text = []
for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
    soup = bs(item.get_content(), 'html.parser')
    text.append(soup.get_text())

full_text = '\n\n'.join(text)

with open(args.output, 'w', encoding='utf-8') as file:
    file.write(full_text)

print(f"Text extracted successfully to {args.output}")



