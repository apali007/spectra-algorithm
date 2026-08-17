import argparse
import json
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import re
import os

MODEL_PATH = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/checkpoints/spectra-chat-checkpoints-v4-prospect/merged_16bit/"

MAX_SEQ_LENGTH = 24576 
MAX_NEW_TOKENS = 6144
TEMPERATURE = 0.1

SYSTEM_PROMPT = (
            "You are SpectraChatModel, an expert mass spectrometry assistant specializing in "
            "de novo peptide sequencing from MS/MS spectra.\n\n"
            "Given a spectrum's precursor mass, precursor charge, and list of fragment ion peaks, "
            "determine the peptide sequence by reasoning step-by-step through complementary y and b ion "
            "fragment ladders:\n"
            "1. First, attempt to derive the sequence using the y-ion ladder (C-terminus to N-terminus). "
            "For each position, identify the fragment peak, verify its mass against the expected "
            "residue mass within tolerance, and compute the residue from the mass difference between "
            "consecutive fragments. If the ladder resolves completely, confirm it against the precursor "
            "mass and stop.\n"
            "2. If gaps remain in the y-ion ladder, attempt the b-ion ladder (N-terminus to C-terminus) "
            "the same way. If the b-ion ladder alone resolves completely, confirm it against the "
            "precursor mass and stop.\n"
            "3. If both ladders have gaps, merge them: for each unresolved b-ion position, check whether "
            "the y-ion ladder resolved that same position, and vice versa.\n"
            "4. If exactly one residue remains unresolved after merging, recover it by subtracting the "
            "known residues and one water molecule from the precursor mass.\n"
            "5. When a terminal residue (the first or last position) has no fragment peak, recover it "
            "from the adjacent known fragment and the precursor mass rather than reporting it as missing.\n"
            # "6. If two or more residues remain unresolved after merging, or the recovered mass does not "
            # "match any residue within tolerance, state clearly that the complete sequence cannot be "
            # "determined, and report the partial sequence with unresolved positions marked.\n\n"
            "Always validate any complete candidate sequence against the precursor mass before reporting "
            "it as final. Show your arithmetic explicitly at each step."
        )


# Thse sample input are from /lclhome/apali007/projects/spectra_algorithm/languagemodel/v2/input_data/test_split.json
sample_input = "Precursor mz: 667.85479 Da, Precursor Charge: 2\nExperimental MS/MS spectrum (m/z, intensity):" \
    "\nm/z 147.11246, intensity 0.31000\nm/z 185.05533, intensity 0.10000\nm/z 195.03891, intensity 0.03000\nm/z 212.06584, intensity 0.14000" \
    "\nm/z 213.05009, intensity 0.12000\nm/z 218.14929, intensity 0.26000\nm/z 230.07646, intensity 1.00000\nm/z 278.11417, intensity 0.02000" \
    "\nm/z 330.10748, intensity 0.02000\nm/z 331.23291, intensity 0.10000\nm/z 348.11713, intensity 0.13000\nm/z 364.72824, intensity 0.01000" \
    "\nm/z 375.13028, intensity 0.02000\nm/z 376.11121, intensity 0.08000\nm/z 393.13873, intensity 0.15000\nm/z 405.76367, intensity 0.09000" \
    "\nm/z 406.26431, intensity 0.02000\nm/z 414.27802, intensity 0.11000\nm/z 419.15717, intensity 0.03000\nm/z 448.76730, intensity 0.02000" \
    "\nm/z 470.30945, intensity 0.02000\nm/z 471.78757, intensity 0.08000\nm/z 473.13153, intensity 0.04000\nm/z 485.22443, intensity 0.03000" \
    "\nm/z 487.33368, intensity 0.49000\nm/z 490.15421, intensity 0.04000\nm/z 491.13788, intensity 0.09000\nm/z 493.73444, intensity 0.02000" \
    "\nm/z 494.22699, intensity 0.11000\nm/z 494.73117, intensity 0.02000\nm/z 502.74188, intensity 0.24000\nm/z 508.16580, intensity 0.18000" \
    "\nm/z 544.31299, intensity 0.02000\nm/z 544.80511, intensity 0.05000\nm/z 545.29236, intensity 0.06000\nm/z 550.77295, intensity 0.02000" \
    "\nm/z 553.32050, intensity 0.44000\nm/z 553.82349, intensity 0.14000\nm/z 558.31573, intensity 0.02000\nm/z 559.28375, intensity 0.04000" \
    "\nm/z 569.37494, intensity 0.03000\nm/z 586.29449, intensity 0.03000\nm/z 586.40192, intensity 0.47000\nm/z 601.17957, intensity 0.02000" \
    "\nm/z 601.82806, intensity 0.06000\nm/z 610.34058, intensity 0.40000\nm/z 618.21326, intensity 0.02000\nm/z 619.19519, intensity 0.05000" \
    "\nm/z 636.22620, intensity 0.12000\nm/z 682.45514, intensity 0.04000\nm/z 687.25146, intensity 0.02000\nm/z 699.48553, intensity 0.43000" \
    "\nm/z 704.29138, intensity 0.02000\nm/z 715.25244, intensity 0.03000\nm/z 731.29834, intensity 0.02000\nm/z 732.28296, intensity 0.06000" \
    "\nm/z 749.30914, intensity 0.12000\nm/z 796.46094, intensity 0.05000\nm/z 810.51990, intensity 0.16000\nm/z 811.52142, intensity 0.04000" \
    "\nm/z 827.54382, intensity 0.55000\nm/z 831.35925, intensity 0.03000\nm/z 848.36932, intensity 0.03000\nm/z 924.56531, intensity 0.05000" \
    "\nm/z 925.55377, intensity 0.08000\nm/z 942.57092, intensity 0.82000\nm/z 959.51770, intensity 0.05000\nm/z 976.49286, intensity 0.01000" \
    "\nm/z 986.47412, intensity 0.02000\nm/z 987.46826, intensity 0.03000\nm/z 1004.47638, intensity 0.24000\nm/z 1073.55994, intensity 0.02000" \
    "\nm/z 1087.62561, intensity 0.03000\nm/z 1088.60034, intensity 0.03000\nm/z 1089.56531, intensity 0.11000\nm/z 1090.55603, intensity 0.02000" \
    "\nm/z 1100.52588, intensity 0.03000\nm/z 1105.63049, intensity 0.29000\nm/z 1106.63342, intensity 0.08000\nm/z 1117.57019, intensity 0.16000" \
    "\nm/z 1118.56238, intensity 0.04000\nm/z 1160.59924, intensity 0.06000\nm/z 1171.57361, intensity 0.03000\nm/z 1188.59583, intensity 0.32000" \
    "\nAnalyze this spectrum and identify the most likely peptide sequence."

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

def predict_from_single_spectrum(model, tokenizer, spectrum_input):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": spectrum_input},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

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
                results = json.load(f)
            processed_ids = {item['spectra_id'] for item in results}
            print(f"Resuming: {len(processed_ids)} already processed")

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


            prediction = predict_from_single_spectrum(
                model, tokenizer, input_text,
            )

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
        default=sample_input,
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
        is_resume = False

        predict_from_batch(model, tokenizer, args.batch, args.output, num_examples, is_resume)
    
    elif args.text:
        
        prediction = predict_from_single_spectrum(model, tokenizer, args.text)
        print(f"OUTPUT:\n{prediction}")
    

if __name__== "__main__":
    main()