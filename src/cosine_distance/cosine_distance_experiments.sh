#!/bin/bash
#PBS -N cosine_distance
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=30gb:mem=30gb
#PBS -o job_logs/outputs
#PBS -e job_logs/errors

percentiles=('5' '10' '15' '20' '25' '30' '40' '50' '60' '70' '75' '80' '85' '90' '95')

export TMPDIR=$SCRATCHDIR

VENVDIR=/storage/brno2/home/maldonj/creativity_detection/venv
DATADIR=/storage/brno2/home/maldonj/creativity_detection

# Calculate cosine distance
$VENVDIR/bin/python3 $DATADIR/src/cosine_distance/cosine_distance.py --input $DATADIR/data/eval/2br02b_parallel.csv --output $DATADIR/experiments/cosine_distance/distances.csv

for perc in ${percentiles[@]};
do
    echo $perc

    # Extract classifications from estimations
    $VENVDIR/bin/python3 $DATADIR/src/eval_utils/extract_classifications.py --input $DATADIR/experiments/cosine_distance/distances.csv --output $DATADIR/experiments/cosine_distance/classifications/2br02b_$perc.csv --percentile $perc
    
    # Evaluate classifications against gold data
    $VENVDIR/bin/python3 $DATADIR/src/eval_utils/evaluate_classifications.py --gold $DATADIR/data/eval/2br02b_gold.csv --candidate $DATADIR/experiments/cosine_distance/classifications/2br02b_$perc.csv --output $DATADIR/experiments/cosine_distance/evaluation.csv
done
