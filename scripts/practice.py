# def mgfoutput(MGF_PATH):
#     with open(MGF_PATH, 'r') as f:
#         for i, line in enumerate(f):
#             if i>=1000:
#                 break
#             print(line,end ='')

# if __name__ == "__main__":
#     MGF_PATH    = "/disk/raptor-array/mshab008/datasets/TMT_Thermo_SRM_Pool/02244_BA10-TMT_Thermo_SRM_Pool_10_01_01-TMT1-1h-R1_annotation.parquet"
#     # Print the top 1000 lines of the mgf file
#     mgfoutput(MGF_PATH)
    

# import pandas as pd
# meta = pd.read_parquet('/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool_meta_data.parquet')
# print('Total rows: ', len(meta))
# print('Unique peptides: ', meta['modified_sequence'].nunique())
# print('Unique scans: ', meta['scan_number'].nunique())

# import glob 

# annot_dir = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool/"
# all_annot_files = glob.glob(annot_dir + "*.parquet")
# annot_data = pd.concat([pd.read_parquet(f) for f in all_annot_files], ignore_index=True)

# print("Total annotation rows: ", len(annot_data))


# import json

# input_path = "/lclhome/apali007/projects/spectra_algorithm/output/reasoning_dataset_7317.json"
# output_path = "/lclhome/apali007/projects/spectra_algorithm/output/reasoning_dataset_7317_positive.json"

# with open(input_path, 'r') as f:
#     data = json.load(f)

# positive = []
# for item in data:
#     if item.get('y_b_match')==True:
#         item['y_match'] = item.pop('y_b_match')
#         positive.append(item)
        
# with open(output_path, 'w') as f:
#     json.dump(positive, f, indent=2)

# print(f"Total: {len(data)}")
# print(f"Positive: {len(positive)}")
# print(f"Saved to {output_path}")

# import json
# from collections import Counter

# with open('/lclhome/apali007/projects/spectra_algorithm/output/reasoning_dataset_6942_positive.json') as f:
#     data = json.load(f)

# peptides = Counter(entry['peptide'] for entry in data)

# print(f'Total entries: {len(data)}')
# print(f'Unique peptides: {len(peptides)}')

# import re
# import pandas as pd
# from collections import Counter

# AA_MASSES = {
#     'A': 71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
#     'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G': 57.02146,
#     'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
#     'M': 131.04049, 'F': 147.06841, 'P': 97.05276,  'S': 87.03203,
#     'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V': 99.06841,
# }

# def is_valid_peptide(peptide):
#     return bool(peptide) and bool(re.fullmatch(r'[A-Z]+', peptide)) and all(aa in AA_MASSES for aa in peptide)

# meta_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool_meta_data.parquet"

# meta_data = pd.read_parquet(meta_data_path)

# meta_data = meta_data[meta_data['modified_sequence'].apply(is_valid_peptide)]

# peptide_counts = Counter(meta_data['modified_sequence'].tolist())

# print(f"Total spectra: {len(meta_data)}")
# print(f"Unique peptides: {len(peptide_counts)}")
# print()
# for peptide, count in peptide_counts.most_common():
#     print(f"{peptide}: {count}")

# import json
# import re

# with open('/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/results-inference-spectra-v4.json') as f:
#     results = json.load(f)

# with open('/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/reasoning_dataset_test.json') as f:
#     test = json.load(f)

# pattern = r'The complete sequence is predicted as ([A-Z]+)'

# none_count = 0
# for d in results:
#     if not re.search(pattern, d.get('prediction', '')):
#         none_count+=1

# print(f'Total: {len(results)}')
# print(f'No valid peptides: {none_count}')
# print(f'Valid predictions: {len(results) - none_count}')
# print()

# mismatches = 0
# for r,t in zip(results, test):
#     if r.get('scan_number') != t.get("scan_number") or r.get('raw_file') != t.get('raw_file'):
#         mismatches += 1

# print(f"Alignment mismatches (scan_number + raw_file): {mismatches}/{len(results)}")

import pandas as pd

annotation_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool/TUM_third_pool_1_01_01_annotation.parquet"
meta_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool_meta_data.parquet"

ann = pd.read_parquet(annotation_data_path)
meta = pd.read_parquet(meta_data_path)

merged = ann.merge(meta, on=['raw_file', 'scan_number'], how='inner')
sapg = merged[
    (merged['peptide_sequence'] == 'SAPGTAQGEER') & (merged['raw_file']=="01812a_GA3-TUM_third_pool_1_01_01-2xIT_2xHCD-1h-R1") 
    & (merged['scan_number']==2639)]

first = sapg.iloc[0]
one_scan = sapg[(sapg['raw_file'] == first['raw_file']) & (sapg['scan_number'] == first['scan_number'])]
                
print(f'Number of rows: {len(one_scan)}')
print(one_scan[['experimental_mass', 'intensity', 'ion_type', 'no']].to_string())

# print('=== Annotation Columns')
# print(ann.columns.tolist())
# print()

# print('=== Meta Columns ===')
# print(meta.columns.tolist())
# print()

# print('=== Annotation Sample ===')
# print(ann.head(2).to_string())
# print()

# print('=== Meta Sample ===')
# print(meta.head(2).to_string())