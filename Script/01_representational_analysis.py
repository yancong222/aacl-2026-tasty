# Date: Sep, 2026
# Author: Yan Cong

import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import os
import math
import csv
import shutil, sys

df = pd.read_csv('data.csv', index_col=0)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
current_model = '_r1qwen7b' # CHANGE NAME HERE

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "meta-llama/Meta-Llama-3-8B"
model_name = "Qwen/Qwen2-7B-Instruct"
model_name = "Qwen/Qwen2-7B"

def get_word_embedding(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, sentence: str, target_word: str):
    try:
        tokenized_output = tokenizer(sentence, return_offsets_mapping=True, return_tensors='pt', add_special_tokens=True)
        input_ids = tokenized_output['input_ids'].to(model.device)
        offset_mapping = tokenized_output['offset_mapping'][0] 

        sentence_lower = sentence.lower()
        target_word_lower = target_word.lower()

        start_char_idx = sentence_lower.find(target_word_lower)
        if start_char_idx == -1:
            print(f"Warning: Target word '{target_word}' not found in sentence.")
            return None
        end_char_idx = start_char_idx + len(target_word_lower)

        target_token_indices = []
        for i, (token_start, token_end) in enumerate(offset_mapping):
            if token_start < end_char_idx and token_end > start_char_idx and token_end > token_start:
                target_token_indices.append(i)

        if not target_token_indices:
            print(f"Warning: No tokens found for target word '{target_word}' using offset mapping.")
            return None

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1].squeeze(0) 

        word_embedding = torch.mean(last_hidden_states[target_token_indices], dim=0)
        return word_embedding

    except Exception as e:
        print(f"Error extracting word embedding for '{target_word}' in '{sentence}': {e}")
        return None

def calculate_cosine_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor):
    if embedding1 is None or embedding2 is None:
        return None
    if embedding1.dim() == 1:
        embedding1 = embedding1.unsqueeze(0)
    if embedding2.dim() == 1:
        embedding2 = embedding2.unsqueeze(0)

    cos_calculator = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    return cos_calculator(embedding1, embedding2).item()



for col_name in ['cos_baseline_smell' + current_model, 'cos_baseline_vision' + current_model, 'cos_baseline_taste' + current_model]:
    if col_name not in df.columns:
        df[col_name] = np.nan 
df.head()

for item_id_val in df['itemd_id'].unique():
    df_item = df[df['itemd_id'] == item_id_val]

    baseline_row = df_item[df_item['condition'] == 'baseline'].iloc[0] if not df_item[df_item['condition'] == 'baseline'].empty else None
    vision_row = df_item[df_item['condition'] == 'vision'].iloc[0] if not df_item[df_item['condition'] == 'vision'].empty else None
    smell_row = df_item[df_item['condition'] == 'smell'].iloc[0] if not df_item[df_item['condition'] == 'smell'].empty else None
    taste_row = df_item[df_item['condition'] == 'taste'].iloc[0] if not df_item[df_item['condition'] == 'taste'].empty else None

    baseline_embedding = None
    vision_embedding = None
    smell_embedding = None
    taste_embedding = None

    if baseline_row is not None:
        content_baseline = baseline_row['begin'] + ' ' + baseline_row['character'] + ' ' + baseline_row['sensory_VP'] + ' ' + baseline_row['sensory_DP_PP'] + ' ' + baseline_row['end'] + ' ' + baseline_row['sensory_VP_s2'] + ' ' + baseline_row['ADV_intensifier'] + ' ' + baseline_row['PPT_JJ']
        target_baseline = baseline_row['PPT_JJ'].strip('.')
        baseline_embedding = get_word_embedding(model, tokenizer, content_baseline, target_baseline)

    if vision_row is not None:
        content_vision = vision_row['begin'] + ' ' + vision_row['character'] + ' ' + vision_row['sensory_VP'] + ' ' + vision_row['sensory_DP_PP'] + ' ' + vision_row['end'] + ' ' + vision_row['sensory_VP_s2'] + ' ' + vision_row['ADV_intensifier'] + ' ' + vision_row['PPT_JJ']
        target_vision = vision_row['PPT_JJ'].strip('.')
        vision_embedding = get_word_embedding(model, tokenizer, content_vision, target_vision)

    if smell_row is not None:
        content_smell = smell_row['begin'] + ' ' + smell_row['character'] + ' ' + smell_row['sensory_VP'] + ' ' + smell_row['sensory_DP_PP'] + ' ' + smell_row['end'] + ' ' + smell_row['sensory_VP_s2'] + ' ' + smell_row['ADV_intensifier'] + ' ' + smell_row['PPT_JJ']
        target_smell = smell_row['PPT_JJ'].strip('.')
        smell_embedding = get_word_embedding(model, tokenizer, content_smell, target_smell)

    if taste_row is not None:
        content_taste = taste_row['begin'] + ' ' + taste_row['character'] + ' ' + taste_row['sensory_VP'] + ' ' + taste_row['sensory_DP_PP'] + ' ' + taste_row['end'] + ' ' + taste_row['sensory_VP_s2'] + ' ' + taste_row['ADV_intensifier'] + ' ' + taste_row['PPT_JJ']
        target_taste = taste_row['PPT_JJ'].strip('.')
        taste_embedding = get_word_embedding(model, tokenizer, content_taste, target_taste)



