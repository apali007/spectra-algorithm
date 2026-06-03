import pandas as pd

df = pd.read_parquet("02244_BA2-TMT_Thermo_SRM_Pool_2_01_01-TMT1-1h-R1_annotation.parquet")

print("Rows:", len(df))
print("Shape (rows, cols):", df.shape)
df.head(1000).to_csv("02244_BA2-TMT_Thermo_SRM_Pool_2_01_01-TMT1-1h-R1_annotation.csv", index=False)
print("Done.")