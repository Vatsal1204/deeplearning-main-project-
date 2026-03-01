# step1_test_dataset.py
import pandas as pd

# Load your dataset
df = pd.read_csv("website_intelligence_dataset_8000.csv")

# Quick check
print("✅ DATASET LOADED SUCCESSFULLY!")
print(f"📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\n🔍 First 3 records:")
print(df.head(3))
print(f"\n📋 Columns: {list(df.columns)}")