# # ================
# # Task: Spectra table from PROSPECT to CSV

# import pandas as pd

# annotation_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool/TUM_third_pool_1_01_01_annotation.parquet"
# annot_data = pd.read_parquet(annotation_data_path)

# rows = annot_data[
#     (annot_data['peptide_sequence'] == 'PQGPPPQEGNK') & 
#     (annot_data['scan_number'] == 4999) & 
#     (annot_data['raw_file'] == '01812a_GA3-TUM_third_pool_1_01_01-DDA-1h-R1')
#     ]

# rows.to_csv("output/PQGPPPQEGNK.csv", index=False)

# # ================


# # ================
# # Task: Print a table

# import pandas as pd

# meta_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool_meta_data.parquet"
# meta_data = pd.read_parquet(meta_data_path)

# # annotation_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool/TUM_third_pool_1_01_01_annotation.parquet"
# # annot_data = pd.read_parquet(annotation_data_path)

# # print(meta_data.head(20))

# # print(meta_data.columns.tolist())

# dupes = meta_data.groupby(['scan_number', 'raw_file']).size()
# dupe_keys = dupes[dupes > 1].index
# sample = meta_data.set_index(['scan_number', 'raw_file']).loc[dupe_keys[:5]]
# print(sample[['modified_sequence', 'andromeda_score']])

# # ================

# # ================
# # Task: Print a 'dict'

# print(dict)
# print(type(dict))

# ================

# # ================
# # Task: check ordinal

# def ordinal(n):
#     """Convert an integer to its ordinal string: 1 -> '1st', 2 -> '2nd', 11 -> '11th', etc."""
#     if 11 <= (n % 100) <= 13:
#         suffix = 'th'
#     else:
#         suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
#     return f"{n}{suffix}"

# for i in [1, 2, 3, 4, 10, 11, 12, 13, 14, 21, 22, 23, 111, 112, 113]:
#     print(i, ordinal(i))

# # ================

# from collections import Counter
# import json

# with open('/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/prospect_reasoning_dataset_train.json') as f:
#     train = json.load(f)
# with open('/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/prospect_reasoning_dataset_test.json') as f:
#     test = json.load(f)

# print("Train case distribution:", Counter(e['case_label'] for e in train))
# print("Test case distribution:", Counter(e['case_label'] for e in test))
# =========================================================

# import json
# import re

# TEST_SET_PATH = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/prospect_reasoning_dataset_test.json'

# def extract_predicted_peptide(prediction):
#     match = re.search(r"Final predicted sequence: ([A-Z]+)\.", prediction)
#     if match:
#         return match.group(1)
#     return None

# def normalize(peptide):
#     return peptide.replace('I', 'L')

# with open(TEST_SET_PATH, 'r') as f:
#     test_set = json.load(f)

# print(f"Total examples in test set: {len(test_set)}")

# total = 0
# match = 0
# none_count = 0

# for entry in test_set:
#     total += 1
#     ground_truth = entry['peptide']
#     # Using the real 'output' field (the ground-truth trace) as a stand-in "prediction"
#     # to confirm the regex correctly extracts from your actual trace format.
#     fake_prediction = entry['output']

#     predicted_peptide = extract_predicted_peptide(fake_prediction)

#     if predicted_peptide is None:
#         none_count += 1
#         print(f"  [NO MATCH] peptide={ground_truth}, output_tail={fake_prediction[-80:]!r}")
#         continue

#     if normalize(predicted_peptide) == normalize(ground_truth):
#         match += 1
#     else:
#         print(f"  [MISMATCH] predicted={predicted_peptide}, ground_truth={ground_truth}")

# print(f"\nTotal: {total}")
# print(f"Extracted (non-None): {total - none_count}")
# print(f"Accuracy: {100*match/total:.1f}%")



# # # =====
# # MAX_SEQ_LENGTH = 12288 (12288 - 11349 (99 percentile) = 939)
# from transformers import AutoTokenizer
# import json

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

# def get_length_stats(path, label):
#     with open(path) as f:
#         data = json.load(f)

#     lengths = []
#     for entry in data:
#         text = entry['input'] + entry['output']  # rough proxy, not exact chat-template length
#         lengths.append(len(tokenizer.encode(text)))

#     lengths_sorted = sorted(lengths)
#     n = len(lengths_sorted)
#     print(f"\n{label} (n={n}):")
#     print(f"  min: {lengths_sorted[0]}")
#     print(f"  mean: {sum(lengths)/n:.0f}")
#     print(f"  median: {lengths_sorted[n//2]}")
#     print(f"  95th pct: {lengths_sorted[int(n*0.95)]}")
#     print(f"  99th pct: {lengths_sorted[int(n*0.99)]}")
#     print(f"  max: {lengths_sorted[-1]}")

#     over_limit = sum(1 for l in lengths if l > 4096)
#     print(f"  examples exceeding 4096 tokens: {over_limit} ({100*over_limit/n:.2f}%)")

# get_length_stats(
#     '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/prospect_reasoning_dataset_train.json',
#     "Train set"
# )

# get_length_stats(
#     '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/prospect_reasoning_dataset_test.json',
#     "Test set"
# )


# ============
# MAX_NEW_TOKENS = 6144 (6144 - 3577 (99 percentile) = 2567)

# from transformers import AutoTokenizer
# import json

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

# def get_output_length_stats(path, label):
#     with open(path) as f:
#         data = json.load(f)

#     lengths = [len(tokenizer.encode(entry['output'])) for entry in data]
#     lengths_sorted = sorted(lengths)
#     n = len(lengths_sorted)

#     print(f"\n{label} (n={n}) — output field only:")
#     print(f"  min: {lengths_sorted[0]}")
#     print(f"  mean: {sum(lengths)/n:.0f}")
#     print(f"  median: {lengths_sorted[n//2]}")
#     print(f"  95th pct: {lengths_sorted[int(n*0.95)]}")
#     print(f"  99th pct: {lengths_sorted[int(n*0.99)]}")
#     print(f"  max: {lengths_sorted[-1]}")

# get_output_length_stats(
#     '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/prospect_reasoning_dataset_train.json',
#     "Train set"
# )
# get_output_length_stats(
#     '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/prospect_reasoning_dataset_test.json',
#     "Test set"
# )

# ============
# Count the input length in test set.

from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

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

def get_real_input_length_stats(path, label):
    with open(path) as f:
        data = json.load(f)
    
    lengths = []
    for entry in data:
        messages = [
            {"role": "system", "content":SYSTEM_PROMPT},
            {"role": "user", "content": entry['input']},
        ]

        tokens = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True)
        lengths.append(len(tokens))

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    print(f"\n{label} (n={n}) - REAL chat-template input length:")
    print(f" min: {lengths_sorted[0]}")
    print(f" mean: {sum(lengths)/n:.0f}")
    print(f" median: {lengths_sorted[n//2]}")
    print(f" 95th pct: {lengths_sorted[int(n*0.95)]}")
    print(f" 99th pct: {lengths_sorted[int(n*0.99)]}")
    print(f" max: {lengths_sorted[-1]}")

# get_real_input_length_stats(
#     '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/prospect_reasoning_dataset_test.json',
#     "Test set"
# )

# get_real_input_length_stats(
#     '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/prospect_reasoning_dataset_train.json',
#     "Train set"
# )

def get_real_full_length_stats(path, label):
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
    print(f"\n{label} (n={n}) - REAL chat-template FULL (system+input+output) length:")
    print(f" min: {lengths_sorted[0]}")
    print(f" mean: {sum(lengths)/n:.0f}")
    print(f" median: {lengths_sorted[n//2]}")
    print(f" 95th pct: {lengths_sorted[int(n*0.95)]}")
    print(f" 99th pct: {lengths_sorted[int(n*0.99)]}")
    print(f" max: {lengths_sorted[-1]}")
    over_limit = sum(1 for l in lengths if l > 12288)
    print(f" examples exceeding 12288: {over_limit} ({100*over_limit/n:.2f}%)")

# get_real_full_length_stats(
#     '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/prospect_reasoning_dataset_train.json',
#     "Train set"
# )

get_real_full_length_stats(
    '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/prospect_reasoning_dataset_test.json',
    "Test set"
)

