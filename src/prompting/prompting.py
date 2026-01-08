"""
    This script prompts LLMs to perform sentence classification
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
import anthropic
import pandas as pd
import time


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help="Path to input file")
parser.add_argument('--output', type=str, required=True, help="Path to output file")
parser.add_argument('--model', type=str, required=True, help="Model name")
parser.add_argument('--model_type', type=str, choices=['local','openai','claude'], default="local", help="Type of API to use. Must correspond to the model provided")
parser.add_argument('--prompt_type', type=str, required=True, choices=['zero-shot','one-shot','few-shot'], help="Type of prompt to use (loads from args.prompts_folder)")
parser.add_argument('--prompts_folder', type=str, required=True, help="Path to folder containing prompt files")
parser.add_argument('--prompt_examples', type=str, choices=['src_only','translation','multi_translation'], default="src_only", help="Amount of examples to add to the prompt")
args = parser.parse_args()



def load_model():
    """Load the specified model"""
    global model
    global tokenizer

    if args.model_type == "local":
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda", dtype=torch.float16)
    elif args.model_type == "openai":
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if openai.api_key==None:
            raise Exception("You must set the environment variable 'OPENAI_API_KEY' to your OpenAI API key")
        model = openai.OpenAI(api_key = openai.api_key)
    elif args.model_type == 'claude':
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key==None:
            raise Exception("You must set the environment variable 'ANTHROPIC_API_KEY' to your Anthropic API key")
        model = anthropic.Anthropic(api_key=api_key)


def generate_response(prompt):
    """Generate a response for a given prompt including the sentence to classify"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if args.model_type == "local":
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_new_tokens=1)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True).split('Output:')[-1].strip()
            elif args.model_type == "openai":
                completion = model.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1,
                    temperature=0
                )
                response = completion.choices[0].message.content.strip()
            elif args.model_type == "claude":
                message = model.messages.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=args.model,
                    max_tokens=1,
                    temperature=0
                )
                response = message.content[0].text.strip()

            # Parse response
            if isinstance(response, str):
                if response == "no" or response == '0':
                    return '0'
                elif response == "yes" or response == '1':
                    return '1'
            
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
    
    return None


def process_input():
    """ Process input from data and prompt files"""
    with open(f"{args.prompts_folder}/{args.prompt_type}.txt", 'r', encoding="utf-8") as file:
        prompt_template = file.read()
    
    input = pd.read_csv(args.input)
    if 'sentence' not in input.columns:
        raise ValueError(f"Input file must contain a 'sentence' column. Columns found: {input.columns.tolist()}")
    
    responses = []
    for i,row in input.iterrows():
        sentence = row['sentence']

        # Construct prompt according to example types
        if args.prompt_examples == 'multi_translation':
            prompt = f"{prompt_template}Sentence: {sentence}\n"
            non_translator_columns = ['segment','sentence','original']
            for translator in [col for col in input.columns if col not in non_translator_columns]:
                translation = row[translator]
                prompt += f"Translation: {translation}\n"
            prompt += f"Output: "
        elif args.prompt_examples == 'translation':
            translation = row['translation']
            prompt = f"{prompt_template}Sentence: {sentence}\nTranslation: {translation}\nOutput: "
        elif args.prompt_examples == 'src_only':
            prompt = f"{prompt_template}Sentence: {sentence}\nOutput: "
            
        classification = generate_response(prompt)
        
        if classification != None:
            responses.append({'segment':row['segment'], 'sentence':sentence, 'classification':classification})

        # Add small delay to avoid API rate limiting
        if args.model_type in ['openai','claude']:
            time.sleep(0.1)

    return responses



if not os.path.exists(args.input):
    raise NotADirectoryError(f"Input file not found: {args.input}")

if not os.path.isdir(args.prompts_folder):
    raise NotADirectoryError(f"Prompt folder not found: {args.prompt_folder}")

load_model()

responses = process_input()

pd.DataFrame(responses).to_csv(args.output, index=False)