"""
    This script fine-tunes RoBERTa for binary classification using LoRA.
    The code assumes that the test data is already stored separately from the rest of the training data.
    The input data is therefore divided into training and validation sets only.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, TaskType
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help="Path to training data file (in CSV format)")
parser.add_argument('--model', type=str, default='roberta-large', help="Base model to finetune. Default: roberta-large")
parser.add_argument('--output_dir', type=str, required=True, help="Path where finetuned model is saved")
parser.add_argument('--max_length', type=int, default=512, help="Maximum sequence length. Default: 512")
parser.add_argument('--batch_size', type=int, default=8, help="Training batch size. Default: 8")
parser.add_argument('--num_epochs', type=int, default=10, help="Number of treining epochs. Default: 3")
parser.add_argument('--learning_rate', type=float, default=5e-6, help="Learning rate. Default: 2e-4")
parser.add_argument('--val_data_size', type=float, default=0.2, help="Proportion of data for valudation set. Default: 0.2")
parser.add_argument('--early_stopping_patience', type=int, default=3, help="Patience for early stopping. Default: 3")
parser.add_argument('--lora_r', type=int, default=16, help="LoRA rank. Default: 16")
parser.add_argument('--lora_alpha', type=int, default=32, help="LoRA alpha scaling. Default: 32")
parser.add_argument('--lora_dropout', type=float, default=0.1, help="LoRA dropout rate. Default: 0.1")
args = parser.parse_args()


class WeightedTrainer(Trainer):
    """Custom weighted trainer to implement a weighted loss function"""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_fn = torch.nn.CrossEntropyLoss(weight = self.class_weights)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom weighted loss function to compensate class imbalance in the data"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits,labels)
        return (loss,outputs) if return_outputs else loss


def tokenize_dataset(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=args.max_length)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0)
    }


# Read data
df = pd.read_csv(args.data)

# Keep only sentence and classification columns
df = df[['sentence','classification']]

# Split data into train and val
train_df, val_df = train_test_split(df, test_size=args.val_data_size, stratify=df['classification'])
train = Dataset.from_pandas(train_df)
val = Dataset.from_pandas(val_df)

# Initialize tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(args.model)
base_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

# Tokenize datasets
tokenized_train = train.map(tokenize_dataset, batched=True, remove_columns=['sentence']).rename_column('classification','labels')
tokenized_val = val.map(tokenize_dataset, batched=True, remove_columns=['sentence']).rename_column('classification','labels')

# Configure LoRA
lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules='all-linear', bias="none", task_type=TaskType.SEQ_CLS)


# Apply LoRA to base model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_peft_model(base_model, lora_config).to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.num_epochs,
    weight_decay=0.01,
    logging_dir=f'{args.output_dir}/logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    save_total_limit=2,
    report_to='none'
)

# Calculate class weights
class_counts = train_df['classification'].value_counts().sort_index()
class_weights = torch.tensor([len(train_df)/(len(class_counts)*count) for count in class_counts])
class_weights = class_weights.to(model.device)

# Create trainer
trainer = WeightedTrainer(
    model=model,
    args = training_args,
    train_dataset = tokenized_train,
    eval_dataset = tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    class_weights = class_weights
)

# Train model
trainer.train()

# Merge LoRA weights and save model
model = model.merge_and_unload()
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
