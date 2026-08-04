import os

MGF_PATH = "/disk/raptor-array/SaeedLab-Data/Mass-Spec/SpectralLibraries/Massive-KB-v1/LIBRARY_CREATION_AUGMENT_LIBRARY_TEST-82c0124b-download_filtered_mgf_library-main.mgf"

try:
    with open(MGF_PATH, "r") as f:
        for i in range(5):
            print(f.readline().strip())
except Exception as e:
    print("Error:", e)