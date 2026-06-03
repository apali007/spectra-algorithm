# import pandas as pd

# # # load meta data
# # meta = pd.read_parquet('./data/TUM_third_pool_meta_data.parquet')

# # print("===META DATA===")
# # print("Shape:", meta.shape)
# # print("Columns:", meta.columns.tolist())
# # print()
# # print(meta.head(3))

# # load one annotation file
# annot = pd.read_parquet('./data/TUM_third_pool/TUM_third_pool_1_01_01_annotation.parquet')

# print("===ANNOTATION===")
# # print("Shape:", annot.shape)
# # print("Columns:", annot.columns.tolist())
# # print()
# # print(annot.head(1000))

# # To CSV file
# annot.head(1000).to_csv('./data/TUM_third_pool_1_01_01_annotation.csv', index=False)

import pandas as pd

# ann = pd.read_csv('/disk/raptor-array/SaeedLab-Data/Mass-Spec/SpectralLibraries/Massive-KB-v1/LIBRARY_CREATION_AUGMENT_LIBRARY_TEST-82c0124b-download_filtered_mgf_library-main.mgf')
# print(ann.head(10).to_string())


df = pd.read_parquet('data/TUM_third_pool_meta_data.parquet'); 
print(df.columns.tolist()); 
print(df.head(2))

