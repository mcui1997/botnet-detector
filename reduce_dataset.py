import pandas as pd

# Read the large dataset
input_file = "training_datasets/UNSW_2018_Iot_Botnet_Dataset_2.csv"
output_file = "training_datasets/UNSW_2018_Iot_Botnet_Dataset_2_half.csv"

# Column names for the dataset
column_names = [
    'pkSeqID', 'stime', 'flgs', 'proto', 'saddr', 'sport', 'daddr', 'dport',
    'pkts', 'bytes', 'state', 'ltime', 'seq', 'dur', 'mean', 'stddev',
    'res_bps_payload', 'res_pps_payload', 'res_bps_ratio', 'res_pps_ratio',
    'ar_bps_payload', 'ar_pps_payload', 'ar_bps_ratio', 'ar_pps_ratio',
    'ar_bps_delta', 'trans_depth', 'response_body_len', 'ct_srv_src',
    'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'category', 'subcategory'
]

# Read dataset
print("Reading dataset...")
data = pd.read_csv(input_file, header=None, names=column_names)
print(f"Original size: {len(data):,} rows")

# Keep only the first half
half_size = len(data) // 3
data_half = data.head(half_size)
print(f"New size: {len(data_half):,} rows")

# Save without headers (same format as original)
print("Saving reduced dataset...")
data_half.to_csv(output_file, index=False, header=False)
print(f"Saved to: {output_file}")