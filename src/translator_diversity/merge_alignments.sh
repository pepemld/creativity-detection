#!/bin/bash
#PBS -N merge_alignments
#PBS -l select=1:ncpus=1
#PBS -o job_logs/outputs
#PBS -e job_logs/errors

books=('1984' 'alice_wonderland' 'dracula' 'frankenstein' 'great_gatsby' 'huckleberry_finn' 'moby_dick' 'pride_prejudice' 'ulysses' 'wuthering_heights')

export TMPDIR=$SCRATCHDIR

VENVDIR=/storage/brno2/home/maldonj/creativity_detection/venv
DATADIR=/storage/brno2/home/maldonj/creativity_detection

for book in ${books[@]};
do
    echo $book
    $VENVDIR/bin/python3 $DATADIR/src/translator_diversity/merge_alignments.py --alignments $DATADIR/data/aligned_books/alignment_files/alignments/$book --book $DATADIR/data/books/$book --output $DATADIR/data/aligned_books/$book.csv
done
