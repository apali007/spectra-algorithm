import argparse
import json
import os
import re
from datetime import datetime
import logging
from collections import Counter
import hashlib

from transformers import set_seed
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from config import Config
from constants import SYSTEM_PROMPT, SAMPLE_INPUT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


# def extract_predicted_peptide(prediction):
#     match = re.search(r"Final predicted sequence: ([A-Z]+)\.", prediction)
#     if match:
#         return match.group(1)
#     return None


# def normalize(peptide):
#     return peptide.replace('I', 'L') if peptide else peptide

# def evaluate(results):
#     """Reports two numbers, since they answer different questions:
#     - attempted accuracy: correctness among spectra the model produced a
#       final sequence for
#     - overall accuracy: correctness over ALL spectra, treating "no final
#       sequence" as an incorrect/unresolved prediction, not a skip.
#     Overall accuracy is the fair number for comparing against Casanovo and
#     other baselines, since it doesn't let "declined to answer" cases quietly
#     inflate the score.
#     """
#     total = len(results)
#     attempted = 0
#     match_count = 0
#     count_pred_none = 0
#     pred_pep_no_extraction = 0

#     for item in results:
#         ground_truth = item.get('ground_truth')
#         prediction = item.get('prediction')

#         if prediction is None:
#             count_pred_none += 1

#         predicted_peptide = extract_predicted_peptide(prediction) if prediction else None

#         if predicted_peptide is None:
#             pred_pep_no_extraction += 1
#             continue

#         attempted += 1
#         if ground_truth and normalize(predicted_peptide) == normalize(ground_truth):
#             match_count += 1

#     logger.info(f"\nTotal spectra: {total}")
#     logger.info(f"Attempted (model produced a final sequence): {attempted} ({100*attempted/total:.1f}% coverage)" if total else "Attempted: 0")

#     if attempted > 0:
#         logger.info(f"Attempted accuracy: {100*match_count/attempted:.1f}%  ({match_count}/{attempted})")
#     else:
#         logger.info("Attempted accuracy: n/a (no predictions attempted)")

#     if total > 0:
#         logger.info(f"Overall accuracy:   {100*match_count/total:.1f}%  ({match_count}/{total})")
#     else:
#         logger.info("Overall accuracy: n/a (empty result set)")

#     if pred_pep_no_extraction > 0:
#         logger.info(f"Prediction cannot be extracted: {100*pred_pep_no_extraction/total:.1f}%  ({pred_pep_no_extraction}/{total})")
#     else:
#         logger.info("Prediction cannot be extracted: n/a (all predictions can be extracted)")

#     if count_pred_none > 0:
#         logger.info(f"Prediction none or empty: {100*count_pred_none/total:.1f}%  ({count_pred_none}/{total})")
#     else:
#         logger.info("Prediction none or empty: n/a (no none or empty prediction)")

#     return {
#         "total": total,
#         "attempted": attempted,
#         "correct": match_count,
#         "attempted_accuracy_pct": (100 * match_count / attempted) if attempted else None,
#         "overall_accuracy_pct": (100 * match_count / total) if total else None,
#     }


def predict_from_single_spectrum(model, tokenizer, cfg: Config, spectrum_input):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": spectrum_input},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    inputs = input_ids.to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    prediction = tokenizer.decode(
        outputs[0][inputs.shape[1]:], skip_special_tokens=True
    )

    return prediction


def predict_from_spectrum_batch(model, tokenizer, cfg: Config, spectrum_inputs):
    """Generates predictions for a LIST of spectra in one forward pass.

    Left-padding is required here (unlike training, which uses right-padding):
    for decoder-only generation, padding must sit before the real tokens so
    that every sequence in the batch ends at the same position and generation
    continues correctly from there. Because of left-padding, every sequence's
    prompt occupies the same number of columns after tokenization, so a single
    slice index (input_len) correctly separates prompt from generated tokens
    for every item in the batch at once.
    """
    texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": spectrum_input},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for spectrum_input in spectrum_inputs
    ]

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.max_seq_length,
    ).to("cuda")

    outputs = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    input_len = encoded["input_ids"].shape[1]
    predictions = [
        tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True)
        for j in range(len(texts))
    ]
    return predictions

def spectra_key(item):
    raw_file = item.get('raw_file')
    scan_number = item.get('scan_number')
    peptide = item.get('peptide')
    return f"{raw_file}::{scan_number}::{peptide}"

def shard_of(key: str, num_shards: int) -> int:
    """
    Deterministic bucket assignment via md5 - 
    Dont use Python's built-in hash() as it can randomize per-process for strings, which would silently reassign
    every spectrum to a different shard on every run.
    """
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_shards

def predict_from_batch(model, tokenizer, cfg: Config, input_batch_path, output_path, num_examples, shard_index, num_shards, seed):
    with open(input_batch_path, 'r') as f:
        data = json.load(f)

    if num_examples is not None:
        data = data[:num_examples]

    keyed_data = [(spectra_key(item), item) for item in data]

    key_counts = Counter(key for key, _ in keyed_data)
    dupes = {key for key, count in key_counts.items() if count > 1}
    if dupes:
        logger.warning(f"{len(dupes)} duplicate spectra_key(s) found - these spectra " \
                       f"may be silently skipped due to key collisions: {list(dupes)[:5]}" \
                       f"{'...' if len(dupes) > 5 else ''}")

    ids_in_scope = {key for key, _ in keyed_data if shard_of(key, num_shards) == shard_index}
    shard_data = [(key, item) for key, item in keyed_data if key in ids_in_scope]

    results = []
    processed_ids = set()

    if output_path and os.path.exists(output_path):
        with open(output_path, 'r') as f:
            loaded_results = json.load(f)

        results = [item for item in loaded_results if spectra_key(item) in ids_in_scope]

        processed_ids = {spectra_key(item) for item in results if item.get("prediction") not in (None,"")}
        failed_count = len(results) - len(processed_ids)
        results = [item for item in results if spectra_key(item) in processed_ids]
        if failed_count:
            logger.info(f"{failed_count} previously-failed spectra will be retried this run.")
        logger.info(f"Found {len(processed_ids)} already-processed spectra in {output_path}, resuming and skipping those.")
    
    remaining_data = [(key, item) for key, item in shard_data if key not in processed_ids]
    logger.info(f"Shard {shard_index}/{num_shards}: {len(shard_data)} spectra assigned, {len(remaining_data)} remaining to process")

    if not remaining_data:
        logger.info("All spectra already have predictions - skipping generation, running evaluation only.")
    else:
        batch_size = cfg.inference_batch_size
        for batch_start in range(0, len(remaining_data), batch_size):
            batch = remaining_data[batch_start:batch_start + batch_size]

            batch_keys = [key for key, _ in batch]
            logger.info(f"Processing spectra {batch_keys} (batch of {len(batch)}) ...")

            input_texts = [inst.get('input', None) for _, inst in batch]

            try:
                predictions = predict_from_spectrum_batch(model, tokenizer, cfg, input_texts)
            except Exception as e:
                logger.error(f"Batch {batch_keys} failed: {type(e).__name__}: {e}")
                predictions = [None] * len(batch)

            for (key, spectra_instance), prediction in zip(batch, predictions):
                results.append({
                    "input": spectra_instance.get('input', None),
                    "prediction": prediction,
                    "ground_truth": spectra_instance.get('peptide', None),
                    "case_label": spectra_instance.get('case_label', None),
                    "scan_number": spectra_instance.get('scan_number', None),
                    "raw_file": spectra_instance.get('raw_file', None),
                })

            if output_path:
                with open(output_path, 'w') as out_f:
                    json.dump(results, out_f, indent=2)


    '''
    Evaluation is intentionally NOT run here when sharding is in use - 
    this function only sees ONE shard's worth of data, so any accuracy
    numbers printed here would silently misrepresent the full test set.
    Run evaluate() seperarely over the merged shard output instead.
    '''
    # metrics = evaluate(results) if results else None
    metrics = None

    if output_path:
        manifest_path = output_path.rsplit(".json", 1)[0] + "_manifest.json"
        manifest = {
            "checkpoint": cfg.output_dir,
            "output_path": output_path,
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "max_seq_length": cfg.max_seq_length,
            "inference_batch_size": cfg.inference_batch_size,
            "input_batch_path": input_batch_path,
            "num_examples_requested": num_examples,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "seed": seed,
            "load_in_4bit": cfg.load_in_4bit,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Saved run manifest to {manifest_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reasoning-corpus model inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a merged_16bit checkpoint directory. No default — "
                              "always specify explicitly to avoid accidentally evaluating "
                              "against a stale model.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch", type=str, default=None)

    # below arguments are not defined in jobs/run_inference.sh
    parser.add_argument("--text", type=str, default=SAMPLE_INPUT)
    parser.add_argument("--num_examples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None,
                         help="Spectra generated per forward pass. Defaults to "
                              "Config.inference_batch_size. Larger batches are "
                              "faster but a single failure inside a batch loses "
                              "predictions for the whole batch — start small "
                              "and only raise it once you've confirmed memory "
                              "headroom on your actual data.")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed for generation, so sampled outputs are "
                        "reproducible run-to-run. Defaults to Config.seed.")
    parser.add_argument("--shard_index", type=int, default=0, 
                        help="Which shard this process handles (0-indexed). Used with " \
                        "--num_shards to split one test set across multiple GPU " \
                        "processes without pre-spliting the file.")
    parser.add_argument("--num_shards", type=int, default=1, 
                    help="Total number of shards. Each spectrum goes to exactly " \
                    "one shard, so running all shards together covers the " \
                    "full test set with no overlap.")
    
    return parser


def main():
    cfg = Config()
    parser = build_argparser()
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
        f"Checkpoint path does not exist: {args.checkpoint}\n"
        "Double-check the path - no fallback is used, to avoid silently "
        "evaluating against the wrong model."
        )

    if args.batch and not os.path.exists(args.batch):
        raise FileNotFoundError(f"--batch file does not exist: {args.batch}")

    cfg.output_dir = args.checkpoint  # repurposed here as "which checkpoint we're evaluating"
    if args.batch_size is not None:
        cfg.inference_batch_size = args.batch_size

    seed = args.seed if args.seed is not None else cfg.seed
    set_seed(seed)

    logger.info("Loading model ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.checkpoint,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
    )

    FastLanguageModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    logger.info("Model loaded.\n")

    if args.batch:
        if not (0 <= args.shard_index < args.num_shards):
            raise ValueError(f"--shard_index ({args.shard_index}) must be in [0, {args.num_shards})")

        predict_from_batch(
            model, tokenizer, cfg,
            args.batch, args.output,
            args.num_examples,
            args.shard_index,
            args.num_shards,
            seed,
        )
    elif args.text:
        try:
            prediction = predict_from_single_spectrum(model, tokenizer, cfg, args.text)
        except Exception as e:
            logger.error(f"Failed on single spectrum: {type(e).__name__}: {e}")
            prediction = None

        logger.info(f"OUTPUT:\n{prediction}")


if __name__ == "__main__":
    main()