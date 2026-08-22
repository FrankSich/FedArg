# client/check_raw_data.py
from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
csv_files = sorted(RAW_DIR.glob("*.csv"))

for file in csv_files:
    print("="*60)
    print(f"📄 File: {file.name}")
    try:
        df = pd.read_csv(file, low_memory=False)
        print(f"🔢 Samples: {len(df)}")
        print(f"🧬 Columns: {list(df.columns)}")
        print("\n❓ Missing values per column:")
        print(df.isna().sum())
        print("\n🎯 Outcome value counts:")
        if "Outcome" in df.columns:
            print(df["Outcome"].value_counts())
            print("\n🎯 Unique Outcome values:", df["Outcome"].unique())
        else:
            print("❌ No 'Outcome' column found")
        print("="*60 + "\n")
    except Exception as e:
        print(f"❌ Error reading {file.name}: {e}\n")
