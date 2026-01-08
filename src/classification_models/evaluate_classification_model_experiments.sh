#!/bin/bash
#PBS -N evaluate_models
#PBS -l select=1:ncpus=1
#PBS -o job_logs/outputs
#PBS -e job_logs/errors

#datasets=('Qwen2.5-7B-Instruct' 'Meta-Llama-3-8B-Instruct' 'Mistral-7B-Instruct-v0.3')
#datasets='bleu_90_clipping'
#datasets=('Qwen2.5-7B-Instruct_few-shot_src_only_clipping' 'Meta-Llama-3-8B-Instruct_few-shot_src_only_clipping' 'Mistral-7B-Instruct-v0.3_few-shot_src_only_clipping')
#datasets='Mistral-7B-Instruct-v0.3_few-shot_src_only_clipping'
#datasets='bleu_90'
#datasets=('Qwen2.5-7B-Instruct_few-shot_src_only' 'Mistral-7B-Instruct-v0.3_few-shot_src_only' 'Meta-Llama-3-8B-Instruct_zero-shot_src_only' 'comet' 'bleurt' 'levenshtein' 'comet,levenshtein' 'bleurt,levenshtein')
datasets=('cosine_distance')


models=('roberta' 'deberta')


VENVDIR=/storage/brno2/home/maldonj/creativity_detection/venv
DATADIR=/storage/brno2/home/maldonj/creativity_detection
cd $DATADIR

for model in ${models[@]};
do
    for dataset in ${datasets[@]};
    do
        $VENVDIR/bin/python3 $DATADIR/src/eval_utils/evaluate_classifications.py --gold $DATADIR/data/eval/2br02b_gold.csv --candidate $DATADIR/experiments/classification_models/classifications/$model\_$dataset.csv --output $DATADIR/experiments/classification_models/evaluation.csv
    done
done