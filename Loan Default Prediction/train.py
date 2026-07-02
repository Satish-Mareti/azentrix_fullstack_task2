import pandas as pd

# Load dataset
df = pd.read_csv("data/loan_default_prediction.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df["default"].value_counts())

print("\nClass Distribution (%):")
print(df["default"].value_counts(normalize=True) * 100)