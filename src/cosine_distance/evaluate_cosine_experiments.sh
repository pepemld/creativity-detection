#!/bin/bash
#PBS -N evaluate_cosine
#PBS -l select=1:ncpus=1
#PBS -o job_logs/outputs
#PBS -e job_logs/errors

percentiles=('5' '10' '15' '20' '25' '30' '40' '50' '60' '70' '75' '80' '85' '90' '95')

VENVDIR=/storage/brno2/home/maldonj/creativity_detection/venv
DATADIR=/storage/brno2/home/maldonj/creativity_detection
cd $DATADIR

for perc in ${percentiles[@]};
do
    $VENVDIR/bin/python3 $DATADIR/src/eval_utils/evaluate_classifications.py --gold $DATADIR/data/eval/2br02b_gold.csv --candidate $DATADIR/experiments/cosine_distance/classifications/2br02b_$perc.csv --output $DATADIR/experiments/cosine_distance/evaluation.csv
done

