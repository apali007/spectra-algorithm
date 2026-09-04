import argparse
import json
import re
from datetime import datetime

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from config import Config
from constants import SYSTEM_PROMPT, SAMPLE_INPUT


def extract_predicted_peptide(prediction):
    match = re.search(r"Final predicted sequence: ([A-Z]+)\.", prediction)
    if match:
        return match.group(1)
    return None


def normalize(peptide):
    return peptide.replace('I', 'L') if peptide else peptide


def evaluate(results):
    """Reports two numbers, since they answer different questions:
    - attempted accuracy: correctness among spectra the model produced a
      final sequence for (what the old script computed)
    - overall accuracy: correctness over ALL spectra, treating "no final
      sequence" as an incorrect/unresolved prediction, not a skip.
    Overall accuracy is the fair number for comparing against Casanovo and
    other baselines, since it doesn't let "declined to answer" cases quietly
    inflate the score.
    """
    total = len(results)
    attempted = 0
    match_count = 0

    for item in results:
        ground_truth = item.get('ground_truth')
        prediction = item.get('prediction')

        predicted_peptide = extract_predicted_peptide(prediction) if prediction else None

        if predicted_peptide is None:
            continue

        attempted += 1
        if ground_truth and normalize(predicted_peptide) == normalize(ground_truth):
            match_count += 1

    print(f"\nTotal spectra: {total}")
    print(f"Attempted (model produced a final sequence): {attempted} ({100*attempted/total:.1f}% coverage)" if total else "Attempted: 0")

    if attempted > 0:
        print(f"Attempted accuracy: {100*match_count/attempted:.1f}%  ({match_count}/{attempted})")
    else:
        print("Attempted accuracy: n/a (no predictions attempted)")

    if total > 0:
        print(f"Overall accuracy:   {100*match_count/total:.1f}%  ({match_count}/{total})")
    else:
        print("Overall accuracy: n/a (empty result set)")

    return {
        "total": total,
        "attempted": attempted,
        "correct": match_count,
        "attempted_accuracy_pct": (100 * match_count / attempted) if attempted else None,
        "overall_accuracy_pct": (100 * match_count / total) if total else None,
    }


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


def predict_from_batch(model, tokenizer, cfg: Config, input_batch_path, output_path, num_examples):
    results = []

    with open(input_batch_path, 'r') as f:
        data = json.load(f)

    if num_examples is not None:
        data = data[:num_examples]

    print(f"{len(data)} spectra to process")

    batch_size = cfg.inference_batch_size
    for batch_start in range(0, len(data), batch_size):
        batch = list(enumerate(data[batch_start:batch_start + batch_size], batch_start + 1))
        batch_ids = [i for i, _ in batch]
        print(f"Processing spectra {batch_ids[0]}-{batch_ids[-1]} (batch of {len(batch)}) ...")

        input_texts = [inst.get('input', None) for _, inst in batch]

        try:
            predictions = predict_from_spectrum_batch(model, tokenizer, cfg, input_texts)
        except Exception as e:
            # A failure here affects the WHOLE batch, not just one spectrum —
            # the trade-off for batched generation's speed.
            print(f"Batch {batch_ids[0]}-{batch_ids[-1]} failed: {type(e).__name__}: {e}")
            predictions = [None] * len(batch)

        for (i, spectra_instance), prediction in zip(batch, predictions):
            results.append({
                "spectra_id": i,
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

    metrics = evaluate(results)

    if output_path:
        manifest_path = output_path.rsplit(".json", 1)[0] + "_manifest.json"
        manifest = {
            "checkpoint": cfg.output_dir,
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "max_seq_length": cfg.max_seq_length,
            "inference_batch_size": cfg.inference_batch_size,
            "input_batch_path": input_batch_path,
            "num_examples_requested": num_examples,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved run manifest to {manifest_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reasoning-corpus model inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a merged_16bit checkpoint directory. No default — "
                              "always specify explicitly to avoid accidentally evaluating "
                              "against a stale model.")
    parser.add_argument("--text", type=str, default=SAMPLE_INPUT)
    parser.add_argument("--batch", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num_examples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None,
                         help="Spectra generated per forward pass. Defaults to "
                              "Config.inference_batch_size. Larger batches are "
                              "faster but a single failure inside a batch loses "
                              "predictions for the whole batch — start small "
                              "and only raise it once you've confirmed memory "
                              "headroom on your actual data.")
    return parser


def main():
    cfg = Config()
    parser = build_argparser()
    args = parser.parse_args()

    cfg.output_dir = args.checkpoint  # repurposed here as "which checkpoint we're evaluating"
    if args.batch_size is not None:
        cfg.inference_batch_size = args.batch_size

    print("Loading model ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.checkpoint,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    print("Model loaded.\n")

    if args.batch:
        predict_from_batch(
            model, tokenizer, cfg,
            args.batch, args.output,
            args.num_examples,
        )
    elif args.text:
        try:
            prediction = predict_from_single_spectrum(model, tokenizer, cfg, args.text)
        except Exception as e:
            print(f"Failed on single spectrum: {type(e).__name__}: {e}")
            prediction = None

        print(f"OUTPUT:\n{prediction}")


if __name__ == "__main__":
    main()