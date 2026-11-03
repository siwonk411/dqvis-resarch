import pandas as pd
from datasets import load_dataset

# 1. Load dataset (loads instantly from your local cache)
print("Loading dataset...")
ds = load_dataset("HIDIVE/DQVis")
df = ds["train"].to_pandas()

print(f"\n==========================================")
print(f"DATASET OVERVIEW: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"==========================================\n")

# 2. Numerical Summary Statistics (mean, std, min, max)
print("--- NUMERICAL SUMMARY STATISTICS ---")
print(df[["expertise", "formality", "spec_key_count"]].describe().to_string())

# 3. Categorical Class Distributions
print("\n--- CHART TYPE DISTRIBUTION ---")
chart_counts = df["chart_type"].value_counts()
chart_pct = df["chart_type"].value_counts(normalize=True) * 100
print(pd.DataFrame({"Count": chart_counts, "Percentage (%)": chart_pct.round(2)}).to_string())

print("\n--- CHART COMPLEXITY DISTRIBUTION ---")
comp_counts = df["chart_complexity"].value_counts()
comp_pct = df["chart_complexity"].value_counts(normalize=True) * 100
print(pd.DataFrame({"Count": comp_counts, "Percentage (%)": comp_pct.round(2)}).to_string())

print("\n--- DATASET SCHEMA DISTRIBUTION ---")
schema_counts = df["dataset_schema"].value_counts()
print(schema_counts.to_string())

# 4. Inspect 10 Diverse Examples
print("\n" + "="*50)
print("--- 10 INDIVIDUAL EXAMPLES ---")
print("="*50)

# Sample 10 items stratified across complexity categories
samples = df.groupby("chart_complexity", group_keys=False).apply(
    lambda g: g.sample(min(len(g), 3), random_state=42)
).head(10)

for i, (_, row) in enumerate(samples.iterrows(), 1):
    print(f"\n[EXAMPLE {i}]")
    print(f"Chart Type       : {row['chart_type']} | Complexity: {row['chart_complexity']}")
    print(f"Expertise / Form : {row['expertise']} / {row['formality']}")
    print(f"Natural Query    : {row['query']}")
    print(f"Base Query       : {row['query_base']}")
    print(f"Dataset Schema   : {row['dataset_schema']}")
    print(f"Spec Snippet     : {str(row['spec'])[:220]}...")