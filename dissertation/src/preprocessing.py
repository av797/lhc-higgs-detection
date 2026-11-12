
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch, os

# Get the dissertation directory
project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the dataset path
data_path = os.path.join(project_directory, "higgs", "data", "higgs.csv")

# Load the dataset
data = pd.read_csv(data_path)

total_signal_weight = data.loc[
    data["Label"] == "s", "Weight"
].sum()

total_background_weight = data.loc[
    data["Label"] == "b", "Weight"
].sum()

print("\nFull dataset weight totals:")
print("Signal:", total_signal_weight)
print("Background:", total_background_weight)


# Remove the unused KaggleSet partition
data = data[data["KaggleSet"] != "u"]

# Separate the original training and testing partitions
training_data = data[data["KaggleSet"] == "t"].copy()
test_data = data[data["KaggleSet"].isin(["b", "v"])].copy()

# Split the original training partition into training and validation
train_data, validation_data = train_test_split(
    training_data,
    test_size=0.2,
    random_state=42,
    stratify=training_data["Label"]
)

# Define the 30 numerical input features
feature_columns = [
    "DER_mass_MMC",
    "DER_mass_transverse_met_lep",
    "DER_mass_vis",
    "DER_pt_h",
    "DER_deltaeta_jet_jet",
    "DER_mass_jet_jet",
    "DER_prodeta_jet_jet",
    "DER_deltar_tau_lep",
    "DER_pt_tot",
    "DER_sum_pt",
    "DER_pt_ratio_lep_tau",
    "DER_met_phi_centrality",
    "DER_lep_eta_centrality",
    "PRI_tau_pt",
    "PRI_tau_eta",
    "PRI_tau_phi",
    "PRI_lep_pt",
    "PRI_lep_eta",
    "PRI_lep_phi",
    "PRI_met",
    "PRI_met_phi",
    "PRI_met_sumet",
    "PRI_jet_num",
    "PRI_jet_leading_pt",
    "PRI_jet_leading_eta",
    "PRI_jet_leading_phi",
    "PRI_jet_subleading_pt",
    "PRI_jet_subleading_eta",
    "PRI_jet_subleading_phi",
    "PRI_jet_all_pt"
]

# Identify features containing -999 placeholders
missing_features = [
    column for column in feature_columns
    if (train_data[column] == -999).any()
]

print("Training features:", len(feature_columns))
print("Features containing -999:", len(missing_features))
print(missing_features)



# Create missing-value indicators
for column in missing_features:
    train_data[column + "_missing"] = (train_data[column] == -999).astype(int)
    validation_data[column + "_missing"] = (validation_data[column] == -999).astype(int)
    test_data[column + "_missing"] = (test_data[column] == -999).astype(int)


medians = {}

for column in missing_features:

    # Calculate the median using valid training values only
    median_value = train_data.loc[train_data[column] != -999, column].median()

    # Store the median for later use
    medians[column] = median_value

    # Replace -999 with the same training median in all datasets
    train_data[column] = train_data[column].replace(-999, median_value)
    validation_data[column] = validation_data[column].replace(-999, median_value)
    test_data[column] = test_data[column].replace(-999, median_value)



# Check that no -999 placeholders remain
print("\nRemaining -999 values:")

for column in missing_features:
    print(column, train_data[column].eq(-999).sum())

# Check the number of indicators created
indicator_columns = [column + "_missing" for column in missing_features]

print("\nNumber of missing indicators:", len(indicator_columns))
print("Total input features:", len(feature_columns) + len(indicator_columns))

# Display the medians
print("\nTraining medians:")
for column, median_value in medians.items():
    print(column, median_value)


scaler = StandardScaler()

# Fit the scaler using training data only
train_data[feature_columns] = scaler.fit_transform(train_data[feature_columns])

# Apply the same scaling to validation and testing data
validation_data[feature_columns] = scaler.transform(validation_data[feature_columns])
test_data[feature_columns] = scaler.transform(test_data[feature_columns])

# Check the standardised training feature means
print("\nStandardised training feature means:")
print(train_data[feature_columns].mean().round(3))

# Check the standardised training feature standard deviations
print("\nStandardised training feature standard deviations:")
print(train_data[feature_columns].std().round(3))

label_mapping = {
    "b": 0,
    "s": 1
}

train_labels = train_data["Label"].map(label_mapping).astype(int)
validation_labels = validation_data["Label"].map(label_mapping).astype(int)
test_labels = test_data["Label"].map(label_mapping).astype(int)

# Check the encoded labels
print("\nTraining label counts:")
print(train_labels.value_counts().sort_index())

print("\nValidation label counts:")
print(validation_labels.value_counts().sort_index())

print("\nTesting label counts:")
print(test_labels.value_counts().sort_index())


# Combine the 30 original features and 11 missing indicators
all_feature_columns = feature_columns + indicator_columns


# Extract the input features and convert them into NumPy arrays
X_train_array = train_data[all_feature_columns].to_numpy()
X_validation_array = validation_data[all_feature_columns].to_numpy()
X_test_array = test_data[all_feature_columns].to_numpy()


# Convert the NumPy arrays into PyTorch tensors
X_train = torch.tensor(X_train_array, dtype=torch.float32)
X_validation = torch.tensor(X_validation_array, dtype=torch.float32)
X_test = torch.tensor(X_test_array, dtype=torch.float32)


# Convert the labels into PyTorch tensors
y_train = torch.tensor(train_labels.to_numpy(), dtype=torch.float32)
y_validation = torch.tensor(validation_labels.to_numpy(), dtype=torch.float32)
y_test = torch.tensor(test_labels.to_numpy(), dtype=torch.float32)

w_train = torch.tensor(train_data["Weight"].to_numpy(), dtype=torch.float32)
w_validation = torch.tensor(validation_data["Weight"].to_numpy(), dtype=torch.float32)
w_test = torch.tensor(test_data["Weight"].to_numpy(), dtype=torch.float32)


# Check the shapes of the feature tensors
print("\nFeature tensor shapes:")
print("Training:", X_train.shape)
print("Validation:", X_validation.shape)
print("Testing:", X_test.shape)


# Check the shapes of the label tensors
print("\nLabel tensor shapes:")
print("Training:", y_train.shape)
print("Validation:", y_validation.shape)
print("Testing:", y_test.shape)


# Check the tensor data types
print("\nTensor data types:")
print("Features:", X_train.dtype)
print("Labels:", y_train.dtype)


print("\nEvent weight shapes:")
print("Training:", w_train.shape)
print("Validation:", w_validation.shape)
print("Testing:", w_test.shape)

print("\nEvent weight ranges:")
print("Training:", w_train.min().item(), w_train.max().item())
print("Validation:", w_validation.min().item(), w_validation.max().item())
print("Testing:", w_test.min().item(), w_test.max().item())

results_directory = os.path.join(project_directory, "higgs", "results")
os.makedirs(results_directory, exist_ok=True)

# Define the file path
save_path = os.path.join(results_directory, "processed_data.pt")

# Store all processed data and preprocessing information
processed_data = {
    "X_train": X_train,
    "y_train": y_train,
    "w_train": w_train,

    "X_validation": X_validation,
    "y_validation": y_validation,
    "w_validation": w_validation,

    "X_test": X_test,
    "y_test": y_test,
    "w_test": w_test,

    "scaler": scaler,
    "medians": medians,
    "feature_columns": all_feature_columns,

    "total_signal_weight": total_signal_weight,
    "total_background_weight": total_background_weight
}

# Save everything into one file
torch.save(processed_data, save_path)

print("\nProcessed data saved successfully!")
print("File location:", save_path)
