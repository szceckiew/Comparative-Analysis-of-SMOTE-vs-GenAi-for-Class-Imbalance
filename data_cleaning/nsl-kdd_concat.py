import pandas as pd

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'class', 'difficulty'
]

FOLDER_LOCATION = '../data/raw/'

print("Loading training dataset...")
df_train = pd.read_csv(FOLDER_LOCATION + 'KDDTrain+.txt', header=None, names=columns)

print("Loading test dataset...")
df_test = pd.read_csv(FOLDER_LOCATION + 'KDDTest+.txt', header=None, names=columns)

print("Concatenating datasets...")
df_full = pd.concat([df_train, df_test], ignore_index=True)

output_filename = FOLDER_LOCATION + 'nsl_kdd_full_dataset.csv'
df_full.to_csv(output_filename, index=False)

print("------------------------------------------------")
print(f"Done! Created file: {output_filename}")
print(f"Row count (Train): {len(df_train)}")
print(f"Row count (Test): {len(df_test)}")
print(f"Total rows: {len(df_full)}")
print("First 5 rows preview:")
print(df_full.head())