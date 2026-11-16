
import torch, os, math
from model import HiggsClassifier
from sklearn.metrics import roc_auc_score, f1_score
from metrics import calculate_ams


project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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