"""Quick exploration of the two CSV data files."""

import pandas as pd

FILES = {
    "f0sye9u908ua3i1r.csv": "Small (~2.5GB)",
    "dqvnyhjezmxxzvjl.csv": "Large (~28GB)",
}

for fname, label in FILES.items():
    print(f"\n{'='*60}")
    print(f"{label}: {fname}")
    print(f"{'='*60}")

    df = pd.read_csv(f"data/{fname}", nrows=1000, low_memory=False)

    print(f"\nShape (first 1000 rows): {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns[:20])}{'...' if len(df.columns) > 20 else ''}")
    print(f"\nDtypes:\n{df.dtypes.value_counts()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nBasic stats:\n{df.describe()}")
    print(f"\nNull counts (top 20):\n{df.isnull().sum().sort_values(ascending=False).head(20)}")
    print(f"\nSample values per column:")
    for col in df.columns[:30]:
        print(f"  {col}: {df[col].dropna().unique()[:5]}")
