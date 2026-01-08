#!/bin/bash
#PBS -N finetune_and_classify
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=50gb:mem=50gb
#PBS -l walltime=48:00:00 
#PBS -o job_logs/outputs
#PBS -e job_logs/errors


VENVDIR=/storage/brno2/home/maldonj/creativity_detection/venv
DATADIR=/storage/brno2/home/maldonj/creativity_detection
cd $DATADIR

datasets=('Qwen2.5-7B-Instruct_few-shot_src_only' 'Mistral-7B-Instruct-v0.3_few-shot_src_only' 'Meta-Llama-3-8B-Instruct_zero-shot_src_only' 'comet' 'levenshtein' 'comet,levenshtein' 'bleurt' 'bleurt,levenshtein' 'cosine_distance')


# RoBERTa
for dataset in ${datasets[@]};
do
    echo $dataset
    $VENVDIR/bin/python3 $DATADIR/src/classification_models/finetune.py --data $DATADIR/data/datasets/$dataset.csv --output_dir $DATADIR/experiments/classification_models/finetuned_models/roberta_$dataset
    $VENVDIR/bin/python3 $DATADIR/src/classification_models/classify.py --input $DATADIR/data/eval/2br02b.csv --output $DATADIR/experiments/classification_models/classifications/roberta_$dataset.csv --model-path $DATADIR/experiments/classification_models/finetuned_models/roberta_$dataset
done


# DeBERTa
for dataset in ${datasets[@]};
do
    echo $dataset
    $VENVDIR/bin/python3 $DATADIR/src/classification_models/finetune.py --data $DATADIR/data/datasets/$dataset.csv --output_dir $DATADIR/experiments/classification_models/finetuned_models/deberta_$dataset --model 'microsoft/deberta-v3-large'
    $VENVDIR/bin/python3 $DATADIR/src/classification_models/classify.py --input $DATADIR/data/eval/2br02b.csv --output $DATADIR/experiments/classification_models/classifications/deberta_$dataset.csv --model-path $DATADIR/experiments/classification_models/finetuned_models/deberta_$dataset
done

