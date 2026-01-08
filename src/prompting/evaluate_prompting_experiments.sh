#!/bin/bash
#PBS -N evaluate_models
#PBS -l select=1:ncpus=1
#PBS -o job_logs/outputs
#PBS -e job_logs/errors

models=('claude-sonnet-4-20250514' 'gpt-4' 'Qwen2.5-7B-Instruct' 'Meta-Llama-3-8B-Instruct' 'Mistral-7B-Instruct-v0.3' 'Llama-3.3-70B-Instruct' 'Llama-3.2-1B' 'Qwen3-1.7B')

prompt_types=('zero-shot_src_only' 'one-shot_src_only' 'few-shot_src_only' 'one-shot_translation' 'few-shot_translation' 'one-shot_multi_translation' 'few-shot_multi_translation')


VENVDIR=/storage/brno2/home/maldonj/creativity_detection/venv
DATADIR=/storage/brno2/home/maldonj/creativity_detection
cd $DATADIR

for model in ${models[@]};
do
    for prompt_type in ${prompt_types[@]};
    do
        $VENVDIR/bin/python3 $DATADIR/src/eval_utils/evaluate_classifications.py --gold $DATADIR/data/eval/2br02b_gold.csv --candidate $DATADIR/experiments/prompting/classifications/$model\_$prompt_type.csv --output $DATADIR/experiments/prompting/evaluation.csv
    done
done