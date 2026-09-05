import argparse
import json
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import re
import os

# MODEL_PATH = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/checkpoints/all_clean_y_ion_case_spectra-chat-checkpoints-prospect/merged_16bit/"
MODEL_PATH = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/checkpoints/system_prompt_30k_all_cases_spectra-chat-checkpoints-prospect/merged_16bit/"


MAX_SEQ_LENGTH = 31210 
MAX_NEW_TOKENS = 6144
TEMPERATURE = 0.1

from constants import (
    AA_MASSES, NEUTRAL_LOSS_MASSES, MASS_PROTON,
    TOLERANCE_BY_ANALYZER, PRECURSOR_TOLERANCE,
    CASE_CLEAN_B_ION_LADDER, CASE_CLEAN_Y_ION_LADDER,
    CASE_COMPLEMENTARY_GAPS, CASE_UNRESOLVABLE_MULTI_GAP,
    CASE_SINGLE_GAP_RESOLVED, CASE_UNRESOLVABLE_SINGLE_GAP_MASS_MISMATCH,
    CASES, SYSTEM_PROMPT, SAMPLE_INPUT
)

TOLERANCE = 0.2


AA_MASS_TABLE_TEXT = "\n".join(
    f"{aa}: {mass:.5f} Da" for aa, mass in AA_MASSES.items()
)

def extract_predicted_peptide(prediction):
    match = re.search(r"Final predicted sequence: ([A-Z]+)\.", prediction)
    if match:
        return match.group(1)
    
    return None
    
def normalize(peptide):
    return peptide.replace('I','L')

def evaluate(results):
    total = 0
    match =0

    for item in results:
        total +=1
        ground_truth = item.get('ground_truth')
        prediction = item.get('prediction')

        predicted_peptide = extract_predicted_peptide(prediction)

        if predicted_peptide is None:
            continue

        if normalize(predicted_peptide) == normalize(ground_truth):
            match +=1

    print(f"\n Total: {total}")
    print(f"Accuracy : {100*match/total:.1f}%")

def predict_from_single_spectrum(i, model, tokenizer, spectrum_input):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": spectrum_input},
    ]

    inputs_ids = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    inputs = inputs_ids.to("cuda")

    outputs = model.generate(
        input_ids = inputs,
        max_new_tokens = MAX_NEW_TOKENS,
        temperature = TEMPERATURE,
        do_sample = True,
        pad_token_id= tokenizer.eos_token_id
    )

    prediction = tokenizer.decode(
        outputs[0][inputs.shape[1]:], skip_special_tokens=True
    )

    return prediction


def predict_from_batch(model, tokenizer, input_batch_path, output_path, num_examples, is_resume):
     
    results = []
    if is_resume:
        processed_ids = set()
        if output_path and os.path.exists(output_path):
            with open(output_path, 'r') as f:
                loaded_results = json.load(f)
            results = [item for item in loaded_results if item['prediction'] is not None]
            processed_ids = {item['spectra_id'] for item in results}
            print(f"Resuming: {len(processed_ids)} already processed")
            print(f"Dropping {len(loaded_results) - len(results)} previously-skipped spectra for retry")

    with open(input_batch_path, 'r') as f:

        data = json.load(f)
        
        for i, spectra_instance in enumerate(data, 1):
    
            if num_examples is not None and i > num_examples:
                break

            if is_resume:
                if i in processed_ids:
                    print(f"Skipping spectrum {i} (already done).")
                    continue
            
            print(f"Processing spectrum {i} ...")

            input_text = spectra_instance.get('input', None)
            ground_truth = spectra_instance.get('peptide', None)
            case_label = spectra_instance.get('case_label', None)
            scan_number = spectra_instance.get('scan_number', None)
            raw_file = spectra_instance.get('raw_file', None)

            try:
                prediction = predict_from_single_spectrum(
                    i, model, tokenizer, input_text,
                )
            except ValueError as e:
                print(f"Skipping spectrum {i}: {e}")
                prediction = None

            results.append({
                "spectra_id": i,
                "input":input_text,
                "prediction":prediction,
                "ground_truth": ground_truth,
                "case_label": case_label,
                "scan_number": scan_number,
                "raw_file": raw_file,
            })

            if output_path:
                with open(output_path,'w') as out_f:
                    json.dump(results, out_f, indent=2)
            
    evaluate(results)

def main():

    parser = argparse.ArgumentParser(
        description="SpectraChatModel Inference",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=MODEL_PATH,
    )

    parser.add_argument(
        "--text",
        type=str,
        default=SAMPLE_INPUT,
    )

    parser.add_argument(
        "--batch",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    print("Loading model ... ")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.checkpoint,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True
    )

    FastLanguageModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, chat_template = "qwen-2.5")
    print("Model loaded.\n")

    if args.batch:
        num_examples = None
        is_resume = True

        predict_from_batch(model, tokenizer, args.batch, args.output, num_examples, is_resume)
    
    elif args.text:

        try:
            prediction = predict_from_single_spectrum(
                "the given", model, tokenizer, args.text
            )
        except ValueError as e:
            print(f"Skipping spectrum the given: {e}")
            prediction = None

        print(f"OUTPUT:\n{prediction}")  
    

if __name__== "__main__":
    main()