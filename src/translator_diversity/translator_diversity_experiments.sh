#!/bin/bash
#PBS -N transl_diversity
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=50gb:mem=50gb
#PBS -l walltime=48:00:00 
#PBS -o job_logs/outputs
#PBS -e job_logs/errors

percentiles=('5' '10' '15' '20' '25' '30' '40' '50' '60' '70' '75' '80' '85' '90' '95')

#metrics=('bleu' 'bertscore' 'comet' 'bleurt' 'ter' 'levenshtein' 'cosine')
#metrics=('bleu' 'bertscore' 'comet' 'bleurt' 'ter' 'levenshtein' 'cosine' 'bertscore,bleurt' 'bertscore,bleu' 'bleurt,levenshtein' 'cosine,bleurt' 'cosine,bleu' 'bleu,bleurt')
#metrics=('comet,cosine' 'comet,bleurt')
#metrics=('comet,levenshtein' 'comet,bleurt' 'bleurt,levenshtein' 'comet,bleurt,levenshtein')
metrics=('cosine,levenshtein' 'comet,bleurt,cosine,levenshtein' 'comet,bleurt,cosine')

export TMPDIR=$SCRATCHDIR
export HF_HOME=/storage/brno2/home/maldonj/.cache/huggingface

VENVDIR=/storage/brno2/home/maldonj/creativity_detection/venv
DATADIR=/storage/brno2/home/maldonj/creativity_detection
BLEURTDIR=/storage/brno2/home/maldonj/creativity_detection/external_tools/BLEURT-20
cd $DATADIR

source $VENVDIR/bin/activate
hf auth login --token $(cat $DATADIR/login_tokens/huggingface.txt)

for metric in ${metrics[@]};
do
    # Generate estimations
    $VENVDIR/bin/python3 $DATADIR/src/translator_diversity/translator_diversity.py --input $DATADIR/data/aligned_books/2br02b.csv --metrics $metric --bleurt_model $BLEURTDIR --output $DATADIR/experiments/translator_diversity/creativity_estimations/2br02b_$metric.csv
    
    for perc in ${percentiles[@]};
    do
        echo $perc $metric

        # Extract classifications from estimations
        $VENVDIR/bin/python3 $DATADIR/src/eval_utils/extract_classifications.py --input $DATADIR/experiments/translator_diversity/creativity_estimations/2br02b_$metric.csv --output $DATADIR/experiments/translator_diversity/classifications/2br02b_$metric\_$perc.csv --percentile $perc
        
        # Evaluate classifications against gold data
        $VENVDIR/bin/python3 $DATADIR/src/eval_utils/evaluate_classifications.py --gold $DATADIR/data/eval/2br02b_gold.csv --candidate $DATADIR/experiments/translator_diversity/classifications/2br02b_$metric\_$perc.csv --output $DATADIR/experiments/translator_diversity/evaluation.csv
    done
done