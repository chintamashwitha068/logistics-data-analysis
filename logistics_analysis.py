import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("../data/delivery_logistics.csv")

# Basic data inspection
print("Dataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Display dataset preview
print("\nDataset Preview:")
print(df.head())
