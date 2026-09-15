import pandas as pd

FOLDER_LOCATION = '../data/raw/'

df_train = pd.read_csv(FOLDER_LOCATION + 'shuttle.trn', sep=r'\s+', header=None)
df_test = pd.read_csv(FOLDER_LOCATION + 'shuttle.tst', sep=r'\s+', header=None)

df_full = pd.concat([df_train, df_test], ignore_index=True)

column_names = [f'A{i}' for i in range(1, 10)] + ['Class']
df_full.columns = column_names

output_filename = FOLDER_LOCATION + 'shuttle_full_dataset.csv'
df_full.to_csv(output_filename, index=False)

print(f"Done. Created file: {output_filename}")
print(f"Row count in training set: {len(df_train)}")
print(f"Row count in test set: {len(df_test)}")
print(f"Total row count: {len(df_full)}")