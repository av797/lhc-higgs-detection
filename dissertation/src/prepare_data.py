
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Get the dissertation directory
project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the dataset path
data_path = os.path.join(project_directory, "higgs", "data", "higgs.csv")

# Load dataset
data = pd.read_csv(data_path)

# Remove unused partition
data = data[data["KaggleSet"] != "u"]

training_data = data[data["KaggleSet"] == "t"].copy()
test_data = data[data["KaggleSet"].isin(["b", "v"])].copy()

# Split training data into training and validation
train_data, validation_data = train_test_split(
    training_data,
    test_size=0.2,
    random_state=42,
    stratify=training_data["Label"]
)

print("Training events:", len(train_data))
print("Validation events:", len(validation_data))
print("Testing events:", len(test_data))

print("\nTraining labels:")
print(train_data["Label"].value_counts())

print("\nValidation labels:")
print(validation_data["Label"].value_counts())

print("\nTesting labels:")
print(test_data["Label"].value_counts())
