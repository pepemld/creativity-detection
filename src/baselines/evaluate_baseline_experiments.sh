#!/bin/bash
#PBS -N evaluate_baselines
#PBS -l select=1:ncpus=1
#PBS -o job_logs/outputs
#PBS -e job_logs/errors

baselines=('all_creative' 'random_1' 'random_2' 'random_3' 'random_4' 'random_5')

VENVDIR=/storage/brno2/home/maldonj/creativity_detection/venv
DATADIR=/storage/brno2/home/maldonj/creativity_detection
cd $DATADIR

for baseline in ${baselines[@]};
do
    $VENVDIR/bin/python3 $DATADIR/src/eval_utils/evaluate_classifications.py --gold $DATADIR/data/eval/2br02b_gold.csv --candidate $DATADIR/experiments/baselines/$baseline.csv --output $DATADIR/experiments/baselines/evaluation.csv
done

