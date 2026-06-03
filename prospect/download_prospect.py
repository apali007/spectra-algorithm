import prospectdataset as prospect

prospect.download_dataset(
    record = 'prospect',
    task = 'all',
    save_directory = './data',
    select_pool = 'third_pool'
)

print("Done!")