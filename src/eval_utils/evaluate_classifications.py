"""
    This script evaluates a candidate classification against a gold standard.
    Matches sentences based on segment IDs
"""

import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gold', type=str, required=True, help="Path to gold standard file")
parser.add_argument('--candidate', type=str, required=True, help="Path to candidate file")
parser.add_argument('--output', type=str, required=True, help="Path to file to append evaluation results. If file does not exist, it is created.")
args = parser.parse_args()


def match_and_align(gold, candidate):
    """Match gold and candidate data based on segment IDs"""
    matched_data = []
    unmatched_gold = []

    for i, gold_row in gold.iterrows():
        gold_segment = str(gold_row['segment'])
        gold_class = gold_row['classification']

        matched = False
        for j, cand_row in candidate.iterrows():
            cand_segment = str(cand_row['segment'])
            cand_class = cand_row['classification']

            if gold_segment==cand_segment or gold_segment in cand_segment.split('~~~'):
                if gold_class != 0 and gold_class != 1:
                    continue
                matched_data.append({
                    'gold_segment':gold_segment,
                    'candidate_segment':cand_segment,
                    'gold_classification':gold_class,
                    'candidate_classification':cand_class
                })
                matched=True
                break
    
        if not matched:
            unmatched_gold.append((gold_segment, gold_row['sentence']))
    
    if len(matched_data)==0:
        raise Exception("No matching segments found between gold and candidate files")

    matched_df = pd.DataFrame(matched_data)

    if unmatched_gold:
        print(f"WARNING: {len(unmatched_gold)} gold segments could not be matched")

    return matched_df

def permutation_test(y_true, y_pred, n_permutations=1000, random_seed=42):
    """Do permutation test to determine if classifications are significantly better than chance"""
    np.random.seed(random_seed)
    
    # Calculate observed accuracy
    observed_accuracy = np.mean(y_true == y_pred)
    
    # Generate null distribution by permuting labels
    null_accuracies = []
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    for _ in range(n_permutations):
        permuted_labels = np.random.permutation(y_true_array)
        perm_accuracy = np.mean(permuted_labels == y_pred_array)
        null_accuracies.append(perm_accuracy)
    
    # Calculate p-value
    null_accuracies = np.array(null_accuracies)
    p_value = np.mean(null_accuracies >= observed_accuracy)
    
    return p_value



def compute_metrics(y_true, y_pred):
    """Compute classification metrics given the gold and predicted labels"""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

    confusion = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = confusion

    tn, fp, fn, tp = confusion.ravel()
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp

    metrics['perm_test_pvalue'] = permutation_test(y_true, y_pred, n_permutations=10000)

    return metrics


def append_to_csv(output, metrics, candidate_name, num_matched):
    """Append the computed metrics as a single row to a larger file including all results"""
    
    row_data = {
        'model': candidate_name,
        'matched_samples': num_matched,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'perm_test_pvalue': metrics['perm_test_pvalue']
    }

    if 'true_positives' in metrics:
        row_data.update({
            'true_positives': metrics['true_positives'],
            'true_negatives': metrics['true_negatives'],
            'false_positives': metrics['false_positives'],
            'false_negatives': metrics['false_negatives'],
        })
    
    row_df = pd.DataFrame([row_data])

    if os.path.exists(output):
        row_df.to_csv(output, mode='a', header=False, index=False)
    else:
        row_df.to_csv(output, mode='w', header=True, index=False)




gold_df = pd.read_csv(args.gold)
candidate_df = pd.read_csv(args.candidate)

matched_df = match_and_align(gold_df, candidate_df)
gold_labels = matched_df['gold_classification'].values
candidate_labels = matched_df['candidate_classification'].values

metrics = compute_metrics(gold_labels, candidate_labels)

candidate_name = os.path.basename(args.candidate)
if candidate_name.endswith('.csv'):
    candidate_name = candidate_name.split('.csv')[0]

append_to_csv(args.output, metrics, candidate_name, len(matched_df))
print(f"Evaluation of {args.candidate} completed and appended to {args.output}")