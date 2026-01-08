#!/bin/bash
#PBS -N dataset_transl_div_colv
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=50gb:mem=50gb
#PBS -l walltime=48:00:00 
#PBS -o job_logs/outputs
#PBS -e job_logs/errors

threshold=-1.2
metric="comet,levenshtein"

books=('1984' 'alice_wonderland' 'dracula' 'great_gatsby' 'huckleberry_finn' 'moby_dick' 'pride_prejudice' 'ulysses' 'wuthering_heights')

export TMPDIR=$SCRATCHDIR
export HF_HOME=/storage/brno2/home/maldonj/.cache/huggingface

VENVDIR=/storage/brno2/home/maldonj/creativity_detection/venv
DATADIR=/storage/brno2/home/maldonj/creativity_detection
BLEURTDIR=/storage/brno2/home/maldonj/creativity_detection/external_tools/BLEURT-20
cd $DATADIR

source $VENVDIR/bin/activate
hf auth login --token $(cat $DATADIR/login_tokens/huggingface.txt)

mkdir $DATADIR/data/datasets/book_creativity_estimations
mkdir $DATADIR/data/datasets/book_classifications
mkdir $DATADIR/data/datasets/book_creativity_estimations/$metric
mkdir $DATADIR/data/datasets/book_classifications/$metric

for book in ${books[@]};
do
    # Generate estimations
    $VENVDIR/bin/python3 $DATADIR/src/translator_diversity/translator_diversity.py --input $DATADIR/data/aligned_books/$book.csv --metrics $metric --bleurt_model $BLEURTDIR --output $DATADIR/data/datasets/book_creativity_estimations/$metric/$book.csv

    # Extract classifications from estimations
    $VENVDIR/bin/python3 $DATADIR/src/eval_utils/extract_classifications.py --input $DATADIR/data/datasets/book_creativity_estimations/$metric/$book.csv --output $DATADIR/data/datasets/book_classifications/$metric/$book\_$percentile.csv --threshold $threshold
done

# Unify all book classifications into the complete dataset
$VENVDIR/bin/python3 $DATADIR/src/classification_models/unify_dataset.py --input_dir $DATADIR/data/datasets/book_classifications/$metric --output $DATADIR/data/datasets/$metric\_$percentile.csv

