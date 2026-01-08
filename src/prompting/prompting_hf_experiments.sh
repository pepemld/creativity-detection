#!/bin/bash
#PBS -N hf_llms_prompting
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=90gb:mem=90gb
#PBS -l walltime=48:00:00 
#PBS -o job_logs/outputs
#PBS -e job_logs/errors

models=('meta-llama/Llama-3.2-1B' 'Qwen/Qwen3-1.7B' 'meta-llama/Meta-Llama-3-8B-Instruct' 'Qwen/Qwen2.5-7B-Instruct' 'mistralai/Mistral-7B-Instruct-v0.3')

prompt_types=('zero-shot' 'one-shot' 'few-shot')

prompt_examples=('multi_translation' 'translation' 'src_only')

export TMPDIR=$SCRATCHDIR
export HF_HOME=/storage/brno2/home/maldonj/.cache/huggingface

VENVDIR=/storage/brno2/home/maldonj/creativity_detection/venv
DATADIR=/storage/brno2/home/maldonj/creativity_detection
cd $DATADIR
source venv/bin/activate

for model in ${models[@]};
do
    model_name=$(basename "$model")
    echo $model_name

    for prompt_type in ${prompt_types[@]};
    do

        for example_type in ${prompt_examples[@]};
        do
            # Only do zero-shot for source only - no examples, so would be the same in all scenarios
            [[ "$prompt_type" == "zero-shot" && "$example_type" != "src_only" ]] && continue

            echo $model_name $prompt_type $example_type
            SECONDS=0

            # Select the correct input according to example type
            if [[ "$example_type" == "multi_translation" ]]; then
                input_file="2br02b_multi_parallel"
            elif [[ "$example_type" == "translation" ]]; then
                input_file="2br02b_parallel"
            else
                input_file="2br02b"
            fi

            # Generate classifications
            $VENVDIR/bin/python3 $DATADIR/src/prompting/prompting.py --input $DATADIR/data/eval/$input_file.csv --output $DATADIR/experiments/prompting/classifications/$model_name\_$prompt_type\_$example_type.csv --model $model --prompt_type $prompt_type --prompts_folder $DATADIR/data/prompts/$example_type --prompt_examples $example_type

            echo "$model $prompt_type completed in $SECONDS seconds \n"
            # Evaluate classifications against gold data
            $VENVDIR/bin/python3 $DATADIR/src/eval_utils/evaluate_classifications.py --gold $DATADIR/data/eval/2br02b_gold.csv --candidate $DATADIR/experiments/prompting/classifications/$model_name\_$prompt_type\_$example_type.csv --output $DATADIR/experiments/prompting/evaluation.csv
        done
    done
    
    # Clear temporary files
    rm -rf $SCRATCHDIR/*
done

