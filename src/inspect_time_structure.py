"""
inspect_time_structure.py – Check temporal ordering in raw datasets.

This script loads the raw CSV files, prints column names, identifies 
potential time-related columns, and samples first/last rows so you can 
decide whether a time‑blind train/test split is required.

Usage: python src/inspect_time_structure.py
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# -------------------------------------------------------------------
# Dataset specifications (names, file names, expected separator)
# -------------------------------------------------------------------
datasets = {
    "bank": {
        "file": RAW_DIR / "bank.csv",
        "sep": ";",
        "time_cols": ["month", "day_of_week"]   # name in raw file (check casing!)
    },
    "ecom": {
        "file": RAW_DIR / "ecom.csv",
        "sep": ",",
        "time_cols": ["Month", "SpecialDay", "Weekend"]  # weekend is boolean, not ordinal
    },
    "telco": {
        "file": RAW_DIR / "telco.csv",
        "sep": ",",
        "time_cols": ["customerID", "tenure"]  # tenure is relative, customerID might have order
    },
}

for name, spec in datasets.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"File: {spec['file']}")
    print(f"{'='*60}")
    
    if not spec["file"].exists():
        print(f"  -> File not found: {spec['file']}")
        continue
    
    # Try loading with the specified separator
    try:
        df = pd.read_csv(spec["file"], sep=spec["sep"])
    except Exception as e:
        print(f"  -> Failed to read: {e}")
        continue
    
    print(f"  Shape: {df.shape}")
    print(f"  Actual columns: {list(df.columns)}")
    
    # Check which of the desired time columns are present
    present_cols = [c for c in spec["time_cols"] if c in df.columns]
    missing = [c for c in spec["time_cols"] if c not in df.columns]
    if missing:
        print(f"  Time columns missing: {missing} (maybe normalised later)")
    if not present_cols:
        print("  No time columns found in this raw file.")
        continue
    
    print(f"  Found time column(s): {present_cols}")
    for col in present_cols:
        print(f"\n  --- {col} ---")
        print(f"    dtype: {df[col].dtype}")
        unique_vals = df[col].nunique()
        print(f"    Unique values: {unique_vals}")
        # Show first / last few entries of that column
        print("    First 10 values:")
        print(df[col].head(10).to_string(index=False))
        print("    Last 10 values:")
        print(df[col].tail(10).to_string(index=False))
        
        # For months: try to detect ordering
        if col.lower() == "month":
            # Check if the months are in chronological order
            month_order = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                           "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
            try:
                numeric_months = df[col].str.lower().map(month_order)
                is_sorted = numeric_months.is_monotonic_increasing
                if is_sorted:
                    print("    -> Month values appear CHRONOLOGICAL (monotonic).")
                else:
                    print("    -> Month values are NOT monotonic – may be shuffled.")
                    # Show first occurrence of each month
                    month_seq = numeric_months.values
                    # find first time each month appears
                    first_occurrence = {}
                    for i, m in enumerate(month_seq):
                        if m not in first_occurrence:
                            first_occurrence[m] = i
                    sorted_months = sorted(first_occurrence.keys())
                    print(f"    First occurrence index of each month: { {m: first_occurrence[m] for m in sorted_months} }")
            except Exception as e:
                print(f"    Could not analyze month order: {e}")
        elif col.lower() == "day_of_week":
            # Days of week might have order, but we just show the sample
            pass
        elif col.lower() == "specialday":
            print("    SpecialDay: often 0 for normal days, higher values for holidays.")
        elif col.lower() == "weekend":
            print("    Weekend: boolean, not an ordinal time axis.")
        elif col.lower() == "customerid":
            # Check if customerID is monotonic (suggesting temporal order)
            try:
                # extract numeric part if it's like '7590-VHVEG'
                numeric_id = df[col].str.extract(r'(\d+)').astype(int)[0]
                if numeric_id.is_monotonic_increasing:
                    print("    -> customerID appears to be increasing (maybe temporal).")
                else:
                    print("    -> customerID is not monotonic.")
            except Exception as e:
                print(f"    Could not test monotonicity of customerID: {e}")
        elif col.lower() == "tenure":
            print("    Tenure is a relative duration, not an absolute timestamp.")
    
    # Quick look at the order of rows: is there a hidden temporal index?
    # If no time cols, we could check whether the dataframe looks shuffled
    if not present_cols:
        print("\n  No explicit time columns – checking if the row order might be temporal...")
        # For telco: try customerID
        pass  # already done above

print("\nDone. Use these observations to decide on a temporal vs. random split.")