import argparse
import json
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import re
import os

# MODEL_PATH = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/checkpoints/spectra-chat-checkpoints-v4/merged_16bit/"
MODEL_PATH = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/checkpoints/spectra-chat-checkpoints-v4-10-OVERLAP/merged_16bit/"
# MODEL_PATH = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/checkpoints/spectra-chat-checkpoints-v4-20-OVERLAP/merged_16bit/"


MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.1

SYSTEM_PROMPT = (
                "You are SpectraChatModel, an expert mass spectrometry assistant. "
                "Given a MS/MS spectrum with precursor mass and charge, identify the peptide sequence "
                "by reasoning step-by-step through the y-ion ladder. "
                "Start by identifying y1 candidates, then extend the y-ion ladder residue by residue of amino acids. "
                "If the ladder becomes ambiguous or no candidates are found, state clearly that the "
                "peptide sequence cannot be determined. "
                "Conclude with the complete predicted peptide sequence if derivable."
            )


# Thse sample input are from /lclhome/apali007/projects/spectra_algorithm/languagemodel/v2/input_data/test_split.json
sample_input = "Precursor mz: 466.25480 Da, Precursor Charge: 2\nExperimental MS\/MS spectrum (m\/z):\nm\/z 120.552579\nm\/z 120.570772\nm\/z 129.065854\nm\/z 129.065854\nm\/z 130.589496\nm\/z 141.065854\nm\/z 155.081504\nm\/z 158.092403\nm\/z 158.104979\nm\/z 159.067540\nm\/z 175.118952\nm\/z 183.076418\nm\/z 183.112804\nm\/z 200.088472\nm\/z 200.102967\nm\/z 200.102968\nm\/z 200.121160\nm\/z 200.613168\nm\/z 201.105176\nm\/z 201.123369\nm\/z 201.593921\nm\/z 211.107718\nm\/z 228.134267\nm\/z 228.134268\nm\/z 228.599204\nm\/z 228.617397\nm\/z 228.631892\nm\/z 228.644468\nm\/z 228.650085\nm\/z 229.123900\nm\/z 271.140082\nm\/z 271.176467\nm\/z 271.623210\nm\/z 271.632089\nm\/z 271.632089\nm\/z 271.635786\nm\/z 285.155731\nm\/z 285.155732\nm\/z 285.173925\nm\/z 285.647740\nm\/z 288.128326\nm\/z 288.203016\nm\/z 349.170525\nm\/z 349.188717\nm\/z 349.197597\nm\/z 349.221406\nm\/z 349.662532\nm\/z 349.680725\nm\/z 358.175807\nm\/z 358.199617\nm\/z 358.208496\nm\/z 358.208496\nm\/z 358.212193\nm\/z 358.221072\nm\/z 382.208495\nm\/z 382.208496\nm\/z 391.715099\nm\/z 399.169667\nm\/z 399.198659\nm\/z 399.235044\nm\/z 409.233656\nm\/z 414.180566\nm\/z 414.245944\nm\/z 414.271096\nm\/z 417.245609\nm\/z 427.229959\nm\/z 457.240524\nm\/z 516.314023\nm\/z 599.249374\nm\/z 599.285760\nm\/z 599.285760\nm\/z 617.296325\nm\/z 647.354508\nm\/z 670.322874\nm\/z 670.322874\nm\/z 670.359259\nm\/z 670.377018\nm\/z 687.313037\nm\/z 687.349423\nm\/z 698.317788\nm\/z 698.354174\nm\/z 704.375972\nm\/z 715.344337\nm\/z 715.391957\nm\/z 715.409715\nm\/z 715.409715\nm\/z 715.417109\nm\/z 715.434867\nm\/z 817.460036\nAnalyze this spectrum and identify the most likely peptide sequence."

# {"input":"Precursor mz: 515.76350 Da, Precursor Charge: 2\nExperimental MS\/MS spectrum (m\/z):\nm\/z 147.112804\nm\/z 169.097154\nm\/z 187.107718\nm\/z 228.099894\nm\/z 235.107719\nm\/z 235.107719\nm\/z 240.134268\nm\/z 258.144832\nm\/z 294.181218\nm\/z 295.140081\nm\/z 327.166296\nm\/z 345.176860\nm\/z 368.160483\nm\/z 378.189771\nm\/z 386.169036\nm\/z 387.195054\nm\/z 393.176861\nm\/z 413.708328\nm\/z 417.172839\nm\/z 434.203410\nm\/z 437.181946\nm\/z 437.181946\nm\/z 441.249632\nm\/z 448.219060\nm\/z 455.192511\nm\/z 457.224342\nm\/z 528.281660\nm\/z 609.266739\nm\/z 627.277304\nm\/z 629.329339\nm\/z 668.340238\nm\/z 686.350803\nm\/z 698.314417\nm\/z 755.372266\nm\/z 767.335881\nm\/z 773.382831\nm\/z 783.367181\nm\/z 785.346446\nm\/z 808.398816\nm\/z 826.409380\nm\/z 844.419945\nm\/z 895.430844\nm\/z 913.441408\nm\/z 931.451973\nAnalyze this spectrum and identify the most likely peptide sequence.",
#  "output":"Identifying y1 candidates from the given spectrum to build the peptide sequence: \nmz peak at 147.11280 Da. Subtracting water (18.01056 Da) and a proton (1.00728) gives a residue mass of 128.09496 Da, consistent with amino acid K.\nThe observed y1 candidates: K\nAttempting y-ion ladder starting with K as the C-terminal amino acid (y1 = 147.11280 Da):\nThe difference between y2 (294.18122 Da) and y1 (147.11280 Da) is 147.06841 Da, identifying the next residue as F.\nThe difference between y3 (441.24963 Da) and y2 (294.18122 Da) is 147.06841 Da, identifying the next residue as F.\nThe difference between y4 (528.28166 Da) and y3 (441.24963 Da) is 87.03203 Da, identifying the next residue as S.\nThe difference between y5 (629.32934 Da) and y4 (528.28166 Da) is 101.04768 Da, identifying the next residue as T.\nThe difference between y6 (686.35080 Da) and y5 (629.32934 Da) is 57.02146 Da, identifying the next residue as G.\nThe difference between y7 (773.38283 Da) and y6 (686.35080 Da) is 87.03203 Da, identifying the next residue as S.\nThe difference between y8 (844.41995 Da) and y7 (773.38283 Da) is 71.03711 Da, identifying the next residue as A.\nThe difference between y9 (931.45197 Da) and y8 (844.41995 Da) is 87.03203 Da, identifying the next residue as S.\nNo peak found for y10. Using precursor mass (515.76350 Da, charge=2), the singly charged precursor mass is 1030.51972 Da, the mz for ion y9 is 931.45197 Da. The difference between them is 99.06774 Da, which is consistent with amino acid V. The complete sequence is predicted as VSASGTSFFK.",
#  "peptide":"VSASGTSFFK",
#  "scan_number":27204,
#  "raw_file":"01812a_GA3-TUM_third_pool_1_01_01-3xHCD-1h-R1",
#  "y_match":true}

# {"input":"Precursor mz: 466.25480 Da, Precursor Charge: 2\nExperimental MS\/MS spectrum (m\/z):\nm\/z 120.552579\nm\/z 120.570772\nm\/z 129.065854\nm\/z 129.065854\nm\/z 130.589496\nm\/z 141.065854\nm\/z 155.081504\nm\/z 158.092403\nm\/z 158.104979\nm\/z 159.067540\nm\/z 175.118952\nm\/z 183.076418\nm\/z 183.112804\nm\/z 200.088472\nm\/z 200.102967\nm\/z 200.102968\nm\/z 200.121160\nm\/z 200.613168\nm\/z 201.105176\nm\/z 201.123369\nm\/z 201.593921\nm\/z 211.107718\nm\/z 228.134267\nm\/z 228.134268\nm\/z 228.599204\nm\/z 228.617397\nm\/z 228.631892\nm\/z 228.644468\nm\/z 228.650085\nm\/z 229.123900\nm\/z 271.140082\nm\/z 271.176467\nm\/z 271.623210\nm\/z 271.632089\nm\/z 271.632089\nm\/z 271.635786\nm\/z 285.155731\nm\/z 285.155732\nm\/z 285.173925\nm\/z 285.647740\nm\/z 288.128326\nm\/z 288.203016\nm\/z 349.170525\nm\/z 349.188717\nm\/z 349.197597\nm\/z 349.221406\nm\/z 349.662532\nm\/z 349.680725\nm\/z 358.175807\nm\/z 358.199617\nm\/z 358.208496\nm\/z 358.208496\nm\/z 358.212193\nm\/z 358.221072\nm\/z 382.208495\nm\/z 382.208496\nm\/z 391.715099\nm\/z 399.169667\nm\/z 399.198659\nm\/z 399.235044\nm\/z 409.233656\nm\/z 414.180566\nm\/z 414.245944\nm\/z 414.271096\nm\/z 417.245609\nm\/z 427.229959\nm\/z 457.240524\nm\/z 516.314023\nm\/z 599.249374\nm\/z 599.285760\nm\/z 599.285760\nm\/z 617.296325\nm\/z 647.354508\nm\/z 670.322874\nm\/z 670.322874\nm\/z 670.359259\nm\/z 670.377018\nm\/z 687.313037\nm\/z 687.349423\nm\/z 698.317788\nm\/z 698.354174\nm\/z 704.375972\nm\/z 715.344337\nm\/z 715.391957\nm\/z 715.409715\nm\/z 715.409715\nm\/z 715.417109\nm\/z 715.434867\nm\/z 817.460036\nAnalyze this spectrum and identify the most likely peptide sequence.",
#  "output":"Identifying y1 candidates from the given spectrum to build the peptide sequence: \nmz peak at 175.11895 Da. Subtracting water (18.01056 Da) and a proton (1.00728) gives a residue mass of 156.10111 Da, consistent with amino acid R.\nThe observed y1 candidates: R\nAttempting y-ion ladder starting with R as the C-terminal amino acid (y1 = 175.11895 Da):\nThe difference between y2 (288.20302 Da) and y1 (175.11895 Da) is 113.08406 Da, identifying the next residue as L.\nThe difference between y3 (417.24561 Da) and y2 (288.20302 Da) is 129.04259 Da, identifying the next residue as E.\nThe difference between y4 (516.31402 Da) and y3 (417.24561 Da) is 99.06841 Da, identifying the next residue as V.\nThe difference between y5 (647.35451 Da) and y4 (516.31402 Da) is 131.04049 Da, identifying the next residue as M.\nThe difference between y6 (704.37597 Da) and y5 (647.35451 Da) is 57.02146 Da, identifying the next residue as G.\nThe difference between y7 (817.46004 Da) and y6 (704.37597 Da) is 113.08406 Da, identifying the next residue as L.\nNo peak found for y8. Using precursor mass (466.25480 Da, charge=2), the singly charged precursor mass is 931.50232 Da, the mz for ion y7 is 817.46004 Da. The difference between them is 114.04228 Da, which is consistent with amino acid N. The complete sequence is predicted as NLGMVELR.",
#  "peptide":"NIGMVEIR",
#  "scan_number":33926,
#  "raw_file":"01812a_GA3-TUM_third_pool_1_01_01-2xIT_2xHCD-1h-R1",
#  "y_match":true}

# {"input":"Precursor mz: 461.77500 Da, Precursor Charge: 2\nExperimental MS\/MS spectrum (m\/z):\nm\/z 147.112804\nm\/z 207.114490\nm\/z 218.149918\nm\/z 245.131825\nm\/z 246.134268\nm\/z 246.134268\nm\/z 303.168308\nm\/z 330.182904\nm\/z 330.184589\nm\/z 331.233982\nm\/z 358.215889\nm\/z 419.228897\nm\/z 478.302396\nm\/z 514.302396\nm\/z 532.312961\nm\/z 533.308209\nm\/z 565.334424\nm\/z 645.397025\nm\/z 660.407923\nm\/z 678.418488\nm\/z 688.402838\nm\/z 688.402838\nm\/z 791.502552\nAnalyze this spectrum and identify the most likely peptide sequence.",
#  "output":"Identifying y1 candidates from the given spectrum to build the peptide sequence: \nmz peak at 147.11280 Da. Subtracting water (18.01056 Da) and a proton (1.00728) gives a residue mass of 128.09496 Da, consistent with amino acid K.\nThe observed y1 candidates: K\nAttempting y-ion ladder starting with K as the C-terminal amino acid (y1 = 147.11280 Da):\nThe difference between y2 (218.14992 Da) and y1 (147.11280 Da) is 71.03711 Da, identifying the next residue as A.\nThe difference between y3 (331.23398 Da) and y2 (218.14992 Da) is 113.08406 Da, identifying the next residue as L.\nThe difference between y4 (478.30240 Da) and y3 (331.23398 Da) is 147.06841 Da, identifying the next residue as F.\nThe difference between y5 (565.33442 Da) and y4 (478.30240 Da) is 87.03203 Da, identifying the next residue as S.\nThe difference between y6 (678.41849 Da) and y5 (565.33442 Da) is 113.08406 Da, identifying the next residue as L.\nThe difference between y7 (791.50255 Da) and y6 (678.41849 Da) is 113.08406 Da, identifying the next residue as L.\nNo peak found for y8. Using precursor mass (461.77500 Da, charge=2), the singly charged precursor mass is 922.54273 Da, the mz for ion y7 is 791.50255 Da. The difference between them is 131.04017 Da, which is consistent with amino acid M. The complete sequence is predicted as MLLSFLAK.",
#  "peptide":"MLISFLAK",
#  "scan_number":44879,
#  "raw_file":"01812a_GA3-TUM_third_pool_1_01_01-3xHCD-1h-R1",
#  "y_match":true}

def extract_predicted_peptide(prediction):
    match = re.search(r"The complete sequence is predicted as ([A-Z]+)\.", prediction)
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
    

# def predict_from_batch(model, tokenizer, input_batch_path, output_path, num_examples=100):
     
#     with open(input_batch_path, 'r') as f:
#         data = [json.loads(line) for line in f if line.strip()]
    
#     results = []

#     for i, item in enumerate(data, 1):

#         if i > num_examples:
#             break
        
#         print(f"Processing spectrum {i}/{len(data)}...")

#         spectra_instance = item.get('input', str(item))
#         ground_truth = item.get('peptide', None)
#         y_match = item.get('y_match', None)

#         prediction = predict_from_single_spectrum(
#             model, tokenizer, spectra_instance,
#         )

#         results.append({
#             "spectra_id": i,
#             "input":spectra_instance,
#             "prediction":prediction,
#             "ground_truth": ground_truth,
#             "y_match": y_match
#         })

#         if output_path:
#             with open(output_path,'w') as f:
#                 json.dump(results, f, indent=2)
        
#     evaluate(results)
    


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
            y_match = spectra_instance.get('y_match', None)
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
                "y_match": y_match,
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
        is_resume = True

        predict_from_batch(model, tokenizer, args.batch, args.output, num_examples, is_resume)
    
    elif args.text:
        
        prediction = predict_from_single_spectrum(model, tokenizer, args.text)
        print(f"OUTPUT:\n{prediction}")
    

if __name__== "__main__":
    main()