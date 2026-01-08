"""
    This script analyzes the diversity between translations by different translators.
    It then calculates a creativity score for each line based on the analysis.
    Finally, each line is classified as creative or not creative based on creativity scores.
"""

import argparse
import torch
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sacrebleu import BLEU, TER
from bert_score import score as bertscore
from comet import download_model, load_from_checkpoint
from bleurt import score as bleurt_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import Levenshtein


# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help="Path to input file")
parser.add_argument('--metrics', type=str, required=True, help="Comma-separated list of metrics to use (no spaces). Supported metrics: bleu,bertscore,comet,bleurt,ter,levenshtein,cosine")
parser.add_argument('--output', type=str, required=True, help="Output path")
parser.add_argument('--percentile', type=int, default=75, help="Percentile threshold for high/low diversity. Default:75")
parser.add_argument('--bleurt_model', type=str, help="Path to the bleurt model that has to be downloaded separately")
args = parser.parse_args()


chosen_metrics = []
translators = []
translations = []
original_text = []

results = []

# Metrics
bleu_scorer = None
ter_scorer = None
comet_model = None
bleurt_scorer = None
sentence_transformer = None


def parse_metrics_argument():
    "Parse the metrics argument string"
    valid_metrics = ['bleu','bertscore','comet','bleurt','ter','levenshtein','cosine']

    for metric in args.metrics.split(','):
        if metric in valid_metrics:
            chosen_metrics.append(metric)
    
    if chosen_metrics == []:
        raise ValueError("No valid metrics specified. Options are: bleu,bertscore,comet,bleurt,ter,levenshtein,cosine")
    

def init_metrics():
    global bleu_scorer
    global ter_scorer
    global comet_model
    global bleurt_scorer
    global sentence_transformer

    if 'bleu' in chosen_metrics:
        bleu_scorer = BLEU(effective_order=True)
    if 'ter' in chosen_metrics:
        ter_scorer = TER()
    if 'comet' in chosen_metrics:
        try:
            comet_model_path = download_model("Unbabel/wmt22-comet-da")
            comet_model = load_from_checkpoint(comet_model_path).to("cuda")
        except Exception:
            raise Exception("Comet model could not be loaded")
    if 'bleurt' in chosen_metrics:
        try:
            bleurt_scorer = bleurt_score.BleurtScorer(args.bleurt_model)
        except Exception:
            raise Exception("Bleurt model could not be loaded. The path to the model must be provided using argument '--bleurt_model', and the model should be downloaded to the path already.")
    if 'cosine' in chosen_metrics:
        try:
            sentence_transformer = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        except Exception:
            raise Exception("Cosine model could not be loaded")


def load_data():
    """Read and store aligned texts"""
    """Columns should always be structured as follows: 'segment', 'original', 'translator_1', ..., 'translator_n', where translator columns are the names of each translator"""
    global translations
    global translators
    global original_text

    data = pd.read_csv(args.input)

    translators = data.columns.tolist()[2:]
    if len(translators)<2:
        raise ImportError("Data must include more than one translation aligned to the original text")

    original_text = data.iloc[:,1]
    translations = data.iloc[:,2:]
    translations.columns = translators


def calculate_bleu(sentences):
    """Calculate average BLEU score across all candidate translations of one sentence"""
    global bleu_scorer

    total = 0.0
    for i, candidate in enumerate(sentences):
        references = [ref for j,ref in enumerate(sentences) if j != i]
        bleu_score = bleu_scorer.sentence_score(str(candidate), references)
        total += bleu_score.score/100.0
    
    return total/len(sentences)
    

def calculate_bertscore(sentences):
    """Calculate average BERTscore score across all candidate translations of one sentence"""

    total = 0.0
    for i, candidate in enumerate(sentences):
        references = [ref for j,ref in enumerate(sentences) if j != i]
        P,R,F1 = bertscore([candidate]*len(references), references, lang='es',model_type='bert-base-multilingual-cased', rescale_with_baseline=True, verbose=False, device="cuda")
        avg = F1.mean().item()
        total += avg

    return total/len(sentences)
    

def calculate_comet(sentences):
    """Calculate average COMET score across all candidate translations of one sentence"""
    global comet_model

    total = 0.0
    count = 0
    for i, candidate in enumerate(sentences):
        references = [ref for j, ref in enumerate(sentences) if j != i]
        for ref in references:
            comet_data = [{'src':'', 'mt':candidate, 'ref':ref}]
            model_output = comet_model.predict(comet_data, batch_size=8, num_workers=0, gpus=1, accelerator="gpu")
            if hasattr(model_output,'scores'):
                scores = model_output.scores
            elif isinstance(model_output, list) and len(model_output>0):
                if hasattr(model_output[0],'score'):
                    scores = [pred.score for pred in model_output]
                else:
                    scores = model_output
            avg = np.mean(scores)
            count += 1
            total += avg

    return total/count
    

def calculate_bleurt(sentences):
    """Calculate average BLEURT score across all candidate translations of one sentence"""
    global bleurt_scorer

    total = 0.0
    for i,candidate in enumerate(sentences):
        references = [ref for j, ref in enumerate(sentences) if j != i]
        scores = bleurt_scorer.score(references=references, candidates=[candidate]*len(references))
        scores = [min(1,max(0,score)) for score in scores]
        avg = np.mean(scores)
        total += avg
    
    return total/len(sentences)
    

def calculate_ter(sentences):
    """Calculate average TER score across all candidate translations of one sentence"""
    global ter_scorer
    
    total = 0.0
    for i,candidate in enumerate(sentences):
        references = [ref for j, ref in enumerate(sentences) if j != i]
        scores = []
        for ref in references:
            score = ter_scorer.sentence_score(candidate,[ref])
            scores.append(min(1,score.score/120))
        avg = np.mean(scores)
        total += avg
    
    return total/len(sentences)
    

def calculate_levenshtein(sentences):
    """Calculate average Levenshtein distance across all candidate translations of one sentence""" 

    total = 0.0
    for i,candidate in enumerate(sentences):
        references = [ref for j, ref in enumerate(sentences) if j != i]
        distances = []
        for ref in references:
            distance = Levenshtein.distance(candidate, ref)
            max_len = max(len(candidate),len(ref))
            normalized_distance = distance/max_len if max_len>0 else 0.0
            distances.append(normalized_distance)
        avg = np.mean(distances)
        total += avg
    
    return total/len(sentences)


def calculate_cosine(sentences):
    """Calculate average cosine distance across all candidate translations of one sentence"""
    global sentence_transformer

    embeddings = sentence_transformer.encode(sentences)

    total = 0.0
    for i in range(len(embeddings)):
        candidate = embeddings[i].reshape(1,-1)
        references = np.concatenate([embeddings[:i],embeddings[i+1:]],axis=0)
        distances = cosine_distances(candidate, references)
        distances = [np.minimum(1,dist/2.0) for dist in distances]
        avg = np.mean(distances)
        total += avg
    
    return total/len(sentences)    


def calculate_metrics(sentences):
    """Calculate scores for all chosen metrics"""

    metrics = {}
    if 'bleu' in chosen_metrics:
        metrics['bleu'] = calculate_bleu(sentences)
    if 'bertscore' in chosen_metrics:
        metrics['bertscore'] = calculate_bertscore(sentences)
    if 'comet' in chosen_metrics:
        metrics['comet'] = calculate_comet(sentences)
    if 'bleurt' in chosen_metrics:
        metrics['bleurt'] = calculate_bleurt(sentences)
    if 'ter' in chosen_metrics:
        metrics['ter'] = calculate_ter(sentences)
    if 'levenshtein' in chosen_metrics:
        metrics['levenshtein'] = calculate_levenshtein(sentences)
    if 'cosine' in chosen_metrics:
        metrics['cosine'] = calculate_cosine(sentences)
    
    return metrics


def calculate_text_statistics(text):
    """Calculate statistics for a text"""
    text = str(text).strip()
    tokens = word_tokenize(text.lower())
    sentences = [s.strip() for s in text.replace('!','.').replace('?','.').split('.') if s.strip()]

    return {
        'char_count': len(text),
        'token_count': len(tokens),
        'sentence_count': len(sentences),
        'avg_word_length': np.mean([len(token) for token in tokens])
    }


def calculate_span_statistics(sentence_id):
    """Calculate statistics for an aligned span"""
    global original_text
    global translations

    original = original_text.iloc[sentence_id]
    transls = translations.iloc[sentence_id].tolist()

    original_stats = calculate_text_statistics(original)

    translation_stats = []
    for i,tr in enumerate(transls):
        stats = calculate_text_statistics(tr)
        stats['translator_name'] = translators[i]
        translation_stats.append(stats)
    
    if translation_stats:
        avg_transl_stats = {
            'avg_char_count': np.mean([s['char_count'] for s in translation_stats]),
            'avg_token_count': np.mean([s['token_count'] for s in translation_stats]),
            'avg_sentence_count': np.mean([s['sentence_count'] for s in translation_stats]),
            'avg_word_length': np.mean([s['avg_word_length'] for s in translation_stats]),
            'std_char_count': np.std([s['char_count'] for s in translation_stats]),
            'std_token_count': np.std([s['token_count'] for s in translation_stats]),
            'min_token_count': min([s['token_count'] for s in translation_stats]),
            'max_token_count': max([s['token_count'] for s in translation_stats]),   
        }
    else:
        avg_transl_stats = {
            'avg_char_count': 0, 'avg_token_count': 0, 'avg_sentence_count': 0,
            'avg_word_length': 0, 'std_char_count': 0, 'std_token_count': 0,
            'min_token_count': 0, 'max_token_count': 0
        }

    length_ratios = []
    if original_stats['token_count']>0:
        for stats in translation_stats:
            if stats['token_count']>0:
                length_ratios.append(stats['token_count']/original_stats['token_count'])
    
    if length_ratios:
        ratio_stats = {
            'avg_length_ratio': np.mean(length_ratios),
            'std_length_ratio': np.std(length_ratios),
            'min_length_ratio': min(length_ratios),
            'max_length_ratio': max(length_ratios),
        }
    else:
        ratio_stats = {
            'avg_length_ratio': 0.0,
            'std_length_ratio': 0.0,
            'min_length_ratio': 0.0,
            'max_length_ratio': 0.0,
        }

    return {
        'sentence_id': sentence_id,
        'original_stats': original_stats,
        'translation_stats': avg_transl_stats,
        'length_ratio_stats': ratio_stats,
        'individual_translation_stats': translation_stats
    }


def analyze_diversity():
    """Analyzes the diversity of each aligned sentence"""

    id_diff = 0
    for id, row in translations.iterrows():
        sentences = row.tolist()
        valid_sentences = [sent for sent in sentences if pd.notna(sent) and str(sent).strip()]
        metrics = calculate_metrics(valid_sentences)
        stats = calculate_span_statistics(id)

        # Find segment span
        segment_len = len(original_text.iloc[id].split('~~~'))
        if segment_len>1:
            segment = id+1+id_diff
            for i in range(segment+1,segment+segment_len):
                segment = str(segment) + '~~~' + str(i)
                id_diff+=1
        else:
            segment = str(id+1+id_diff)

        result = {
                'sentence_id': id,
                'segment': segment,
                **metrics,
                **stats,
        }
        results.append(result)


def calculate_creativity_scores():
    """Calculates creativity score of an aligned sentence by combining metrics while converting similarity metrics into distance metrics"""
    """Assumes all scores have been normalized to 0-1 range, and further normalizes the score distributions by quantile normalization"""
    all_metric_scores = []
    for span in results:
        metric_scores = []
        for metric in chosen_metrics:
            if metric in ['bleu','comet','bleurt','bertscore']:
                metric_scores.append(1 - span[metric])
            elif metric in ['ter','cosine','levenshtein']:
                metric_scores.append(span[metric])
        all_metric_scores.append(metric_scores)
        
        #span['creativity_score'] = np.mean(metric_scores)
    

    # Z-score standardization
    all_metric_scores = np.array(all_metric_scores)
    metric_means = np.mean(all_metric_scores, axis=0)
    metric_stds = np.std(all_metric_scores, axis=0)
    standardized_metrics = (all_metric_scores - metric_means) / (metric_stds + 1e-10)

    for i,span in enumerate(results):
        span['creativity_score'] = np.mean(standardized_metrics[i,:])

   

def format_output():
    """Reorders the output columns"""
    global results

    output = []
    for span in results:
        output_span = {}
        output_span['sentence_id'] = span['sentence_id']
        output_span['segment'] = span['segment']
        output_span['creativity_score'] = span['creativity_score']

        for metric in chosen_metrics:
            if metric in span:
                output_span[metric] = span[metric]
        
        og_text = original_text.iloc[span['sentence_id']]
        output_span['sentence'] = og_text
        
        transls = translations.iloc[span['sentence_id']].tolist()
        for tr, translator in zip(transls, translators):
            output_span[translator] = tr
        
        for stat_category in ['original_stats','translation_stats','length_ratio_stats']:
            for k,v in span[stat_category].items():
                output_span[f"{stat_category}_{k}"] = v
        
        output.append(output_span)
        
    return output


parse_metrics_argument()
print(f"Chosen metrics: {', '.join(chosen_metrics)}")

init_metrics()
print("Metrics initiated")

load_data()
print(f"Loaded {len(translations)} sentences from {len(translators)} translators")

analyze_diversity()

calculate_creativity_scores()

output = format_output()

# Save output to CSV
output_df = pd.DataFrame(output)
output_df = output_df.sort_values('creativity_score',ascending=False)
output_df.to_csv(args.output, index=False)