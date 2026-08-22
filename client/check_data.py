import pandas as pd
from pathlib import Path

# 🔁 CHANGE THIS to where your hospital CSV files are stored
DATA_DIR = Path("data/cleaned")  # e.g. "dataset", "csv_files", etc.

# 🔁 CHANGE THIS to your label column name
LABEL_COLUMN = "Outcome"

def inspect_dataset(csv_path):
    print("\n" + "="*60)
    print(f"📄 File: {csv_path.name}")
    print("="*60)

    df = pd.read_csv(csv_path)

    print(f"🔢 Samples: {len(df)}")
    print(f"🧬 Features (including label): {df.shape[1]}")

    print("\n📌 Columns:")
    print(list(df.columns))

    print("\n🧪 Data types:")
    print(df.dtypes)

    print("\n❓ Missing values per column:")
    print(df.isnull().sum())

    if LABEL_COLUMN not in df.columns:
        print(f"\n❌ ERROR: Label column '{LABEL_COLUMN}' NOT FOUND")
        return

    print("\n🎯 Outcome value counts:")
    print(df[LABEL_COLUMN].value_counts())

    print("\n🎯 Unique outcome values:")
    print(df[LABEL_COLUMN].unique())

    if df[LABEL_COLUMN].nunique() < 2:
        print("\n🔥 WARNING: ONLY ONE CLASS PRESENT IN THIS DATASET!")

def main():
    csv_files = list(DATA_DIR.glob("*.csv"))

    if not csv_files:
        print("❌ No CSV files found in:", DATA_DIR)
        return

    for csv_file in csv_files:
        inspect_dataset(csv_file)

if __name__ == "__main__":
    main()
