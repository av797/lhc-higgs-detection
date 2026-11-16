
import torch, os
from model import HiggsClassifier
from sklearn.metrics import roc_auc_score, f1_score
from metrics import calculate_ams
import json, shutil
from datetime import datetime


project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#creates new folder automatically for each experiment
experiments_directory = os.path.join(project_directory, "experiments")
os.makedirs(experiments_directory, exist_ok=True)

experiment_name = "experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")

experiment_directory = os.path.join(experiments_directory, experiment_name)
os.makedirs(experiment_directory)

print("Experiment directory:", experiment_directory)

data_path = os.path.join(project_directory, "higgs", "results", "processed_data.pt")
model_path = os.path.join(project_directory, "higgs", "results", "best_model.pth")

data = torch.load(
    data_path,
    map_location="cpu",
    weights_only=False
)

checkpoint = torch.load(
    model_path,
    map_location="cpu",
    weights_only=False
)

# Save a copy of the tested model checkpoint
experiment_model_path = os.path.join(experiment_directory, "model.pth")
shutil.copy2(model_path, experiment_model_path)

print("Tested model checkpoint saved successfully!")

# Save model and training configurations
experiment_config = {
    "model_config": checkpoint["model_config"],
    "training_config": checkpoint["training_config"]
}

config_path = os.path.join(experiment_directory, "config.json")

with open(config_path, "w") as file:
    json.dump(experiment_config, file, indent=4)

print("Experiment configuration saved successfully!")

print("Test data and best model loaded successfully!")
print("Best model epoch:", checkpoint["epoch"])
print("Best validation AMS:", checkpoint["best_val_ams"])
print("Best threshold:", checkpoint["best_threshold"])

model = HiggsClassifier()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
best_threshold = checkpoint["best_threshold"]

print("Trained neural network restored successfully!")

X_test = data["X_test"]
y_test = data["y_test"]

print("\nTest data loaded:")
print("Test features:", X_test.shape)
print("Test labels:", y_test.shape)

with torch.no_grad():

    outputs = model(X_test)

    probabilities = torch.sigmoid(outputs).squeeze(1)

print("Predictions generated successfully!")
print("Number of predictions:", len(probabilities))

test_auc = roc_auc_score(y_test.numpy(), probabilities.numpy())
print("\nTest ROC-AUC:", round(test_auc, 4))

w_test = data["w_test"].double()

signal_mask = y_test == 1
background_mask = y_test == 0

signal_factor = data["total_signal_weight"] / w_test[signal_mask].sum().item()
background_factor = data["total_background_weight"] / w_test[background_mask].sum().item()

w_test_ams = w_test.clone()
w_test_ams[signal_mask] *= signal_factor
w_test_ams[background_mask] *= background_factor

print("\nRenormalised test weights:")
print("Signal:", w_test_ams[signal_mask].sum().item())
print("Background:", w_test_ams[background_mask].sum().item())

predicted_classes = (probabilities >= best_threshold).int()
test_f1 = f1_score(y_test.numpy(), predicted_classes.numpy())

test_ams = calculate_ams(
    y_test,
    probabilities,
    w_test_ams,
    best_threshold
)

print("\nFinal Test Results:")
print("Test ROC-AUC:", round(test_auc, 4))
print("Test F1-score:", round(test_f1, 4))
print("Test AMS:", round(test_ams, 4))
print("Classification threshold:", best_threshold)

# Save experiment evaluation metrics
experiment_metrics = {
    "best_epoch": checkpoint["epoch"],
    "best_validation_ams": checkpoint["best_val_ams"],
    "classification_threshold": best_threshold,

    "test_metrics": {
        "roc_auc": test_auc,
        "f1_score": test_f1,
        "ams": test_ams
    },

    "test_data": {
        "number_of_events": len(y_test),
        "number_of_signal_events": signal_mask.sum().item(),
        "number_of_background_events": background_mask.sum().item()
    },

    "test_event_weights": {
        "signal_weight": w_test_ams[signal_mask].sum().item(),
        "background_weight": w_test_ams[background_mask].sum().item()
    }
}

metrics_path = os.path.join(experiment_directory, "metrics.json")

with open(metrics_path, "w") as file:
    json.dump(experiment_metrics, file, indent=4)

print("\nExperiment metrics saved successfully!")

# Generate experiment report
report = f"""# Experiment Report

## Experiment Information

- Experiment: {experiment_name}
- Best epoch: {checkpoint["epoch"]}
- Best validation AMS: {checkpoint["best_val_ams"]:.4f}

## Model Architecture

- Architecture: {experiment_config["model_config"]["architecture"]}
- Input features: {experiment_config["model_config"]["input_features"]}
- Output features: {experiment_config["model_config"]["output_features"]}
- Hidden layers: {experiment_config["model_config"]["number_of_hidden_layers"]}
- Total parameters: {experiment_config["training_config"]["number_of_parameters"]}

## Training Configuration

- Optimiser: {experiment_config["training_config"]["optimizer"]["name"]}
- Learning rate: {experiment_config["training_config"]["optimizer"]["parameters"][0]["learning_rate"]}
- Loss function: {experiment_config["training_config"]["loss_function"]}
- Batch size: {experiment_config["training_config"]["batch_size"]}
- Epochs: {experiment_config["training_config"]["epochs"]}

## Test Results

- ROC-AUC: {test_auc:.4f}
- F1-score: {test_f1:.4f}
- AMS: {test_ams:.4f}
- Classification threshold: {best_threshold}

## Test Dataset

- Total events: {len(y_test)}
- Signal events: {signal_mask.sum().item()}
- Background events: {background_mask.sum().item()}
- Signal event weight: {w_test_ams[signal_mask].sum().item():.4f}
- Background event weight: {w_test_ams[background_mask].sum().item():.4f}
"""

report_path = os.path.join(experiment_directory, "report.md")

with open(report_path, "w") as file:
    file.write(report)

print("Experiment report saved successfully!")