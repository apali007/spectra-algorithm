def mgfoutput(MGF_PATH):
    with open(MGF_PATH, 'r') as f:
        for i, line in enumerate(f):
            if i>=1000:
                break
            print(line,end ='')

if __name__ == "__main__":
    MGF_PATH    = "/disk/raptor-array/mshab008/datasets/TMT_Thermo_SRM_Pool/02244_BA10-TMT_Thermo_SRM_Pool_10_01_01-TMT1-1h-R1_annotation.parquet"
    # Print the top 1000 lines of the mgf file
    mgfoutput(MGF_PATH)
    

