import pandas as pd
import glob

annotation_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool"
meta_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool_meta_data.parquet"

# # CASE_CLEAN_Y_ION_LADDER
# raw_file = "01812a_GA3-TUM_third_pool_1_01_01-DDA-1h-R1"
# scan_number = 34341
# peptide = "DNYDQLVRIAK"

# # CASE_CLEAN_B_ION_LADDER
# raw_file = "01812a_GA3-TUM_third_pool_1_01_01-DDA-1h-R1"
# scan_number = 10564
# peptide = "RALEELHQK"

# # CASE_COMPLEMENTARY_GAPS
# raw_file = "01812a_GA3-TUM_third_pool_1_01_01-DDA-1h-R1"
# scan_number = 31699
# peptide = "RDLEINSVNSK"

# # CASE_SINGLE_GAP_RESOLVED
# raw_file = "01812a_GA3-TUM_third_pool_1_01_01-DDA-1h-R1"
# scan_number = 21305
# peptide = "SSNLMEHK"

# CASE_UNRESOLVABLE_SINGLE_GAP_MASS_MISMATCH
# raw_file = "01812a_GA3-TUM_third_pool_1_01_01-DDA-1h-R1"
# scan_number = 33899
# peptide = "VLTTYSWR"

# CASE_UNRESOLVABLE_MULTI_GAP
# raw_file = "01829a_GA3-TUM_third_pool_3_01_01-DDA-1h-R1"
# scan_number = 37547
# peptide = "RSDGTWGTEAGPGWAPWPR"


# load metadata
meta_data = pd.read_parquet(meta_data_path)

row = meta_data[
    (meta_data["raw_file"] == raw_file) &
    (meta_data["scan_number"] == scan_number) &
    (meta_data["modified_sequence"] == peptide)
]

print(row[["raw_file", "scan_number", "modified_sequence", "precursor_mz", "precursor_charge"]])

# load annoations, all pools
annotation_files = sorted(glob.glob(f"{annotation_data_path}/TUM_third_pool_*_01_01_annotation.parquet"))

annot_data_matches = pd.concat(
    [pd.read_parquet(
        f,
        filters=[("raw_file", "==", raw_file), ("scan_number", "==", scan_number), ("peptide_sequence", "==", peptide)]
    ) 
    for f in annotation_files],
    ignore_index = True
)

annot_data_matches.to_csv(f"/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/{peptide}_{scan_number}_annotations.csv", index=False)
print(f"wrote {len(annot_data_matches)} rows")
