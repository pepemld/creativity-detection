#!/bin/bash
#PBS -N api_llms_prompting
#PBS -l select=1:ncpus=1:mem=50gb
#PBS -o job_logs/outputs
#PBS -e job_logs/errors

models=('claude-sonnet-4-20250514' 'gpt-4')

prompt_types=('zero-shot' 'one-shot' 'few-shot')

prompt_examples=('multi_translation' 'translation' 'src_only')

export TMPDIR=$SCRATCHDIR

VENVDIR=/storage/brno2/home/maldonj/creativity_detection/venv
DATADIR=/storage/brno2/home/maldonj/creativity_detection
cd $DATADIR

export ANTHROPIC_API_KEY=$(cat $DATADIR/login_tokens/anthropic.txt)
export OPENAI_API_KEY=$(cat $DATADIR/login_tokens/openai.txt)

for model in ${models[@]};
do
    model_name=$(basename "$model")
    echo $model_name

    if [[ "$model" == "gpt-4" ]]
    then
        model_type=openai
    else
        model_type=claude
    fi

    for prompt_type in ${prompt_types[@]};
    do

        for example_type in ${prompt_examples[@]};
        do
            # Only do zero-shot for source only - no examples, so would be the same in all scenarios
            [[ "$prompt_type" == "zero-shot" && "$example_type" != "src_only" ]] && continue

            echo $model_name $prompt_type with translation
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
            $VENVDIR/bin/python3 $DATADIR/src/prompting/prompting.py --input $DATADIR/data/eval/$input_file.csv --output $DATADIR/experiments/prompting/classifications/$model_name\_$prompt_type\_$example_type.csv --model $model --prompt_type $prompt_type --prompts_folder $DATADIR/data/prompts/$example_type --prompt_examples $example_type --model_type $model_type

            echo "$model $prompt_type with translation completed in $SECONDS seconds \n"
            # Evaluate classifications against gold data
            $VENVDIR/bin/python3 $DATADIR/src/eval_utils/evaluate_classifications.py --gold $DATADIR/data/eval/2br02b_gold.csv --candidate $DATADIR/experiments/prompting/classifications/$model_name\_$prompt_type\_$example_type.csv --output $DATADIR/experiments/prompting/evaluation.csv
        done
    done
done

