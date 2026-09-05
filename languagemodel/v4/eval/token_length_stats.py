# ============
# Measures real tokenized length (using the actual SYSTEM_PROMPT and model
# tokenizer) of a reasoning-corpus dataset, to check for examples that would
# get silently truncated by SFTConfig's max_length during training, or by
# max_new_tokens during inference.
#
# Imports SYSTEM_PROMPT and settings from src/ rather than keeping a local
# copy — a stale local copy would silently measure against the WRONG (e.g.
# shorter, pre-analyzer-aware-tolerance) prompt and understate real lengths.
#
# Usage:
#   python eval/token_length_stats.py --input data/processed/40k_..._train.json --mode all

import argparse
import json
import os
import sys

from transformers import AutoTokenizer

# src/ is a sibling directory of eval/ — add it to the import path so we can
# pull the real, current SYSTEM_PROMPT and Config rather than duplicating them.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from constants import SYSTEM_PROMPT
from config import Config


def get_full_length_stats(tokenizer, cfg, path, label):
    print(f"\npath: {path}")
    with open(path) as f:
        data = json.load(f)

    lengths = []
    for entry in data:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": entry['input']},
            {"role": "assistant", "content": entry['output']},
        ]
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        lengths.append(len(tokens))

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    print(f"\n{label} (n={n}) - FULL (system+input+output) length:")
    print(f" min: {lengths_sorted[0]}")
    print(f" mean: {sum(lengths)/n:.0f}")
    print(f" median: {lengths_sorted[n//2]}")
    print(f" 95th pct: {lengths_sorted[int(n*0.95)]}")
    print(f" 99th pct: {lengths_sorted[int(n*0.99)]}")
    print(f" max: {lengths_sorted[-1]}")
    over_limit = sum(1 for l in lengths if l > cfg.max_seq_length)
    print(f" examples exceeding max_seq_length ({cfg.max_seq_length}): {over_limit} ({100*over_limit/n:.2f}%)")


def get_input_length_stats(tokenizer, cfg, path, label):
    print(f"\npath: {path}")
    with open(path) as f:
        data = json.load(f)

    lengths = []
    for entry in data:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": entry['input']},
            {"role": "assistant", "content": ""},
        ]
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        lengths.append(len(tokens))

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    print(f"\n{label} (n={n}) - INPUT (system+user) length:")
    print(f" min: {lengths_sorted[0]}")
    print(f" mean: {sum(lengths)/n:.0f}")
    print(f" median: {lengths_sorted[n//2]}")
    print(f" 95th pct: {lengths_sorted[int(n*0.95)]}")
    print(f" 99th pct: {lengths_sorted[int(n*0.99)]}")
    print(f" max: {lengths_sorted[-1]}")


def get_output_length_stats(tokenizer, cfg, path, label):
    print(f"\npath: {path}")
    with open(path) as f:
        data = json.load(f)

    lengths = []
    for entry in data:
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": entry['output']},
        ]
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        lengths.append(len(tokens))

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    print(f"\n{label} (n={n}) - OUTPUT (assistant) length:")
    print(f" min: {lengths_sorted[0]}")
    print(f" mean: {sum(lengths)/n:.0f}")
    print(f" median: {lengths_sorted[n//2]}")
    print(f" 95th pct: {lengths_sorted[int(n*0.95)]}")
    print(f" 99th pct: {lengths_sorted[int(n*0.99)]}")
    print(f" max: {lengths_sorted[-1]}")
    over_limit = sum(1 for l in lengths if l > cfg.max_new_tokens)
    print(f" examples exceeding max_new_tokens ({cfg.max_new_tokens}): {over_limit} ({100*over_limit/n:.2f}%)")


def main():
    cfg = Config()

    parser = argparse.ArgumentParser(description="Measure real tokenized length of a reasoning-corpus dataset")
    parser.add_argument("--input", required=True, help="Path to the dataset JSON to measure.")
    parser.add_argument("--label", default=None, help="Display label. Defaults to the input filename.")
    parser.add_argument("--mode", choices=["full", "input", "output", "all"], default="all",
                         help="Which length stats to compute. 'full' checks against max_seq_length "
                              "(training truncation risk), 'output' checks against max_new_tokens "
                              "(inference truncation risk).")
    parser.add_argument("--model_id", default=cfg.model_id,
                         help="Tokenizer to use. Defaults to Config.model_id (the actual model being trained).")
    args = parser.parse_args()

    label = args.label or os.path.basename(args.input)

    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print(f"max_seq_length (from Config): {cfg.max_seq_length}")
    print(f"max_new_tokens (from Config): {cfg.max_new_tokens}")

    if args.mode in ("full", "all"):
        get_full_length_stats(tokenizer, cfg, args.input, label)
    if args.mode in ("input", "all"):
        get_input_length_stats(tokenizer, cfg, args.input, label)
    if args.mode in ("output", "all"):
        get_output_length_stats(tokenizer, cfg, args.input, label)


if __name__ == "__main__":
    main()