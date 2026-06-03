mgf = '/disk/raptor-array/SaeedLab-Data/Mass-Spec/SpectralLibraries/Massive-KB-v1/LIBRARY_CREATION_AUGMENT_LIBRARY_TEST-82c0124b-download_filtered_mgf_library-main.mgf'
counts = []
current_count = 0
in_ions = False
with open(mgf) as f:
    for line in f:
        line = line.strip()
        if line == 'BEGIN IONS':
            in_ions = True
            current_count = 0
        elif line == 'END IONS':
            counts.append(current_count)
            in_ions = False
            if len(counts) >= 1000000:
                break
        elif in_ions and line and line[0].isdigit():
            current_count += 1

print(f'Spectra checked: {len(counts)}')
print(f'Min peaks: {min(counts)}')
print(f'Max peaks: {max(counts)}')
print(f'Avg peaks: {sum(counts)/len(counts):.1f}')
print(f'Spectra with >200 peaks: {sum(1 for c in counts if c > 200)}')
print(f'Spectra with <50 peaks: {sum(1 for c in counts if c < 50)}')