#!/bin/bash
#PBS -N evaluate_translator_diversity
#PBS -l select=1:ncpus=1
#PBS -o job_logs/outputs
#PBS -e job_logs/errors

percentiles=('5' '10' '15' '20' '25' '30' '40' '50' '60' '70' '75' '80' '85' '90' '95')

metrics=('bleu' 'bertscore' 'comet' 'bleurt' 'ter' 'levenshtein' 'cosine' 'bertscore,bleurt' 'bertscore,bleu' 'bleurt,levenshtein' 'cosine,bleurt' 'cosine,bleu' 'bleu,bleurt')
#metrics=('bertscore,bleu' 'bleurt,levenshtein' 'cosine,bleurt' 'cosine,bleu' 'bleu,bleurt')

VENVDIR=/storage/brno2/home/maldonj/creativity_detection/venv
DATADIR=/storage/brno2/home/maldonj/creativity_detection
cd $DATADIR

for metric in ${metrics[@]};
do
    for perc in ${percentiles[@]};
    do
        $VENVDIR/bin/python3 $DATADIR/src/eval_utils/evaluate_classifications.py --gold $DATADIR/data/eval/2br02b_gold.csv --candidate $DATADIR/experiments/translator_diversity/classifications/2br02b_$metric\_$perc.csv --output $DATADIR/experiments/translator_diversity/evaluation.csv
    done
done
