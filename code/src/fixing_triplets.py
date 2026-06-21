import pandas as pd
from pathlib import Path

path = Path("../data/processed/triplets.csv")
df = pd.read_csv(path, on_bad_lines="skip")   # skip broken rows temporarily
df.to_csv(path, index=False, quoting=1)        # quoting=1 = QUOTE_ALL, wraps every field in quotes
print(f"Re-saved {len(df)} rows")