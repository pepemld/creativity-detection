#!/bin/bash
#PBS -N dataset_prompting
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=50gb:mem=50gb
#PBS -l walltime=48:00:00 
#PBS -o job_logs/outputs
#PBS -e job_logs/errors

model='meta-llama/Meta-Llama-3-8B-Instruct'
#model='Qwen/Qwen2.5-7B-Instruct'
#model='mistralai/Mistral-7B-Instruct-v0.3'
model_name=$(basename "$model")

example_type='src_only'

#prompt_type='few-shot'
prompt_type='zero-shot'

export TMPDIR=$SCRATCHDIR
export HF_HOME=/storage/brno2/home/maldonj/.cache/huggingface

VENVDIR=/storage/brno2/home/maldonj/creativity_detection/venv
DATADIR=/storage/brno2/home/maldonj/creativity_detection
cd $DATADIR

$VENVDIR/bin/python3 $DATADIR/src/prompting/prompting.py --input $DATADIR/data/books/original_sentences.csv --output $DATADIR/data/datasets/$model_name\_$prompt_type\_$example_type.csv --model $model --prompt_type $prompt_type --prompts_folder $DATADIR/data/prompts/$example_type

