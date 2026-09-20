# Date: Sep, 2026
# Author: Yan Cong

import os, re, json, math, csv, shutil, sys
import pandas as pd
import numpy as np
from numpy import mean, std
import random

df_catch = pd.read_csv('data.csv')
df_exp   = pd.read_csv('data.csv')


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def generate_once(prompt: str, max_new_tokens=MAX_NEW_TOKENS,
                  temperature=TEMPERATURE, top_p=0.9, top_k=TOP_K):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_ids  = outputs[0][inputs["input_ids"].shape[-1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return gen_text.strip()


def sample_predictions(prompt: str, n=N_SAMPLES,
                       max_new_tokens=MAX_NEW_TOKENS,
                       temperature=TEMPERATURE, top_k=TOP_K):
    return [generate_once(prompt, max_new_tokens=max_new_tokens,
                          temperature=temperature, top_k=top_k)
            for _ in range(n)]

def build_prompt(context: str, options: dict) -> str:

    valid_letters = "/".join(options.keys())          
    opts_text = "\n".join(f"{k} {v}" for k, v in options.items())
    return (
        "Read the following sentences and answer a multiple-choice question. \n\n"
        f"passage: \n{context}\n\n"
        f"choose one without explaining: \n{opts_text}\n\n"
        f"Only output {valid_letters}: "
    )

def build_prompt_base(context, options):
    opts_text = "\n".join(f"{k} {v}" for k, v in options.items())
    valid = "/".join(options.keys())
    return (
        f"Passage: {context}"
    )

def randomize_options(option_values: list) -> dict:
    labels = list("ABC"[:len(option_values)])
    shuffled = option_values[:]
    random.shuffle(shuffled)
    options_dict   = dict(zip(labels, shuffled))
    label_to_orig  = {label: option_values.index(val) + 1
                      for label, val in zip(labels, shuffled)}
    return options_dict, label_to_orig


def extract_choice(text: str, valid="ABC"):
    pattern = rf'\b([{valid}])\b'
    m = re.search(pattern, text)
    if m:
        return m.group(1)
    m = re.search(rf'([{valid}])', text)
    return m.group(1) if m else None

def extract_choice_base(text: str, valid="ABC"):
    text = text.upper()
    m = re.search(rf'\b([{valid}])\b', text)
    if m:
        return m.group(1)
    m = re.search(rf'([{valid}])', text)
    return m.group(1) if m else None

def majority_vote(choices, valid="ABC"):
    valid_choices = [c for c in choices if c in list(valid)]
    if not valid_choices:
        return None, 0.0, {}
    counts      = Counter(valid_choices)
    most_common = counts.most_common()
    if len(most_common) >= 2 and most_common[0][1] == most_common[1][1]:
        winner = None
        conf   = most_common[0][1] / len(valid_choices)
    else:
        winner = most_common[0][0]
        conf   = most_common[0][1] / len(valid_choices)
    return winner, conf, dict(counts)

def run_experiment(df, prefix, context_cols, question_cols,
                   option_cols, csv_path, row_filter=None):
    valid_letters = "".join(list("ABC"[:len(option_cols)]))
    df[prefix + '_choice']     = pd.Series(dtype='object')
    df[prefix + '_confidence'] = pd.Series(dtype='float64')
    df[prefix + '_counts']     = pd.Series(dtype='object')
    df[prefix + '_label_map']  = pd.Series(dtype='object')

    for index, row in df.iterrows():
        if row_filter is not None and not row_filter(index):
            continue

        context  = ' '.join(str(row[c]) for c in context_cols)
        question = ' '.join(str(row[c]) for c in question_cols)

        option_values              = [str(row[c]) for c in option_cols]
        options_dict, label_to_orig = randomize_options(option_values)

        prompt          = build_prompt(context + ' ' + question, options_dict)
        raw_predictions = sample_predictions(prompt)
        choices         = [extract_choice(pred, valid=valid_letters)
                           for pred in raw_predictions]
        winner, confidence, counts = majority_vote(choices, valid=valid_letters)

        df.at[index, prefix + '_choice']     = winner
        df.at[index, prefix + '_confidence'] = confidence
        df.at[index, prefix + '_counts']     = json.dumps(counts)
        df.at[index, prefix + '_label_map']  = json.dumps(label_to_orig)
    return df



def load_model(model_name: str):
    global tokenizer, model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available()
                                               else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model     = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        quantization_config=bnb_config
    )
    model.eval()
    print(f"Loaded: {model_name}")


load_model('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
current_model = 'r1qwen7b'

run_experiment(df_catch,
               prefix        = f'catchtrial_{current_model}',
               context_cols  = CATCH_CONTEXT_COLS,
               question_cols = CATCH_QUESTION_COLS,
               option_cols   = CATCH_OPTION_COLS,
               csv_path      = CATCH_CSV,
               row_filter    = lambda i: i > 0)

run_experiment(df_exp,
               prefix        = f'exp1_{current_model}',
               context_cols  = EXP1_CONTEXT_COLS,
               question_cols = QUESTION_COLS_EXP,
               option_cols   = OPTION_COLS_AB,
               csv_path      = EXP_CSV,
               row_filter    = lambda i: i > 0)

run_experiment(df_exp,
               prefix        = f'exp2_{current_model}',
               context_cols  = EXP2_CONTEXT_COLS,
               question_cols = QUESTION_COLS_EXP,
               option_cols   = OPTION_COLS_AB,
               csv_path      = EXP_CSV,
               row_filter    = lambda i: i > -1)

run_experiment(df_exp,
               prefix        = f'exp3_{current_model}',
               context_cols  = EXP3_CONTEXT_COLS,
               question_cols = QUESTION_COLS_EXP,
               option_cols   = OPTION_COLS_ABC,
               csv_path      = EXP_CSV,
               row_filter    = lambda i: i > -1)
