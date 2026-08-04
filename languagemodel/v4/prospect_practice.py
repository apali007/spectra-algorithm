import pandas as pd

annotation_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool/TUM_third_pool_1_01_01_annotation.parquet"
annot_data = pd.read_parquet(annotation_data_path)

rows = annot_data[annot_data['peptide_sequence'] == 'LNTEVTTWRTK']

rows.to_csv("output/LNTEVTTWRTK.csv", index=False)
