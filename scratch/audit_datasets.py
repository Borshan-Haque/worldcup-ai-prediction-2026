import pandas as pd
import os
import glob

data_dir = "data"
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

for csv_path in sorted(csv_files):
    name = os.path.basename(csv_path)
    print(f"\n========================================\nAUDITING DATASET: {name}\n========================================")
    try:
        df = pd.read_csv(csv_path)
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print("Columns and Types:")
        for col in df.columns:
            null_count = df[col].isnull().sum()
            print(f"  - {col}: {df[col].dtype} ({null_count} missing values)")
        print("First 3 rows:")
        print(df.head(3))
    except Exception as e:
        print(f"Error reading {name}: {e}")
