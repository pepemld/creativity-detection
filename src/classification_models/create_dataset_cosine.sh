#!/bin/bash
#PBS -N dataset_cosine
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=50gb:mem=50gb
#PBS -l walltime=48:00:00 
#PBS -o job_logs/outputs
#PBS -e job_logs/errors

threshold=0.24

books=('1984' 'alice_wonderland' 'dracula' 'great_gatsby' 'huckleberry_finn' 'moby_dick' 'pride_prejudice' 'ulysses' 'wuthering_heights')

export TMPDIR=$SCRATCHDIR
export HF_HOME=/storage/brno2/home/maldonj/.cache/huggingface

VENVDIR=/storage/brno2/home/maldonj/creativity_detection/venv
DATADIR=/storage/brno2/home/maldonj/creativity_detection
cd $DATADIR

source $VENVDIR/bin/activate
hf auth login --token $(cat $DATADIR/login_tokens/huggingface.txt)

mkdir $DATADIR/data/datasets/book_creativity_estimations
mkdir $DATADIR/data/datasets/book_classifications
mkdir $DATADIR/data/datasets/book_creativity_estimations/cosine_distance
mkdir $DATADIR/data/datasets/book_classifications/cosine_distance

for book in ${books[@]};
do
    # Generate estimations
    $VENVDIR/bin/python3 $DATADIR/src/classification_models/cosine_distance.py --input $DATADIR/data/parallel_books/$book.csv --output $DATADIR/data/datasets/book_creativity_estimations/cosine_distance/$book.csv

    # Extract classifications from estimations
    $VENVDIR/bin/python3 $DATADIR/src/eval_utils/extract_classifications.py --input $DATADIR/data/datasets/book_creativity_estimations/cosine_distance/$book.csv --output $DATADIR/data/datasets/book_classifications/cosine_distance/$book.csv --threshold $threshold
done

# Unify all book classifications into the complete dataset
$VENVDIR/bin/python3 $DATADIR/src/classification_models/unify_dataset.py --input_dir $DATADIR/data/datasets/book_classifications/cosine_distance --output $DATADIR/data/datasets/cosine_distance.csv

