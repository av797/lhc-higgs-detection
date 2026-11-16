import torch, os, time, math
from torch.utils.data import TensorDataset, DataLoader
from model import HiggsClassifier
from metrics import calculate_ams
from sklearn.metrics import roc_auc_score, f1_score

torch.manual_seed(42)

def get_model_config(model):

    layers = []

    for layer in model.network:

        if isinstance(layer, torch.nn.Linear):
            layers.append({
                "type": "Linear",
                "input_features": layer.in_features,
                "output_features": layer.out_features
            })

        elif isinstance(layer, torch.nn.ReLU):
            layers.append({
                "type": "ReLU"
            })

        else:
            layers.append({
                "type": layer.__class__.__name__
            })

    linear_layers = [
        layer for layer in layers
        if layer["type"] == "Linear"
    ]

    return {
        "architecture": model.__class__.__name__,
        "layers": layers,
        "number_of_linear_layers": len(linear_layers),
        "number_of_hidden_layers": len(linear_layers) - 1,
        "input_features": linear_layers[0]["input_features"],
        "output_features": linear_layers[-1]["output_features"]
    }


def get_training_config(model, optimizer, criterion, train_loader, validation_loader, num_epochs):

    optimizer_config = {
        "name": optimizer.__class__.__name__,
        "parameters": []
    }

    for group in optimizer.param_groups:

        optimizer_config["parameters"].append({
            "learning_rate": group["lr"],
            "weight_decay": group["weight_decay"],
            "betas": group.get("betas"),
            "eps": group.get("eps")
        })

    return {
        "optimizer": optimizer_config,
        "loss_function": criterion.__class__.__name__,
        "batch_size": train_loader.batch_size,
        "validation_batch_size": validation_loader.batch_size,
        "epochs": num_epochs,
        "number_of_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "training_shuffle": train_loader.sampler.__class__.__name__,
        "validation_shuffle": validation_loader.sampler.__class__.__name__,
        "drop_last": train_loader.drop_last,
        "random_seed": 42,
    }

def find_best_ams_threshold(labels, predictions, weights):

    labels = torch.tensor(labels, dtype=torch.int64)
    predictions = torch.tensor(predictions, dtype=torch.float64)

    best_ams = 0.0
    best_threshold = 0.0

    for threshold in [i / 100 for i in range(1, 100)]:

        ams = calculate_ams(
            labels,
            predictions,
            weights,
            threshold
        )

        if ams > best_ams:
            best_ams = ams
            best_threshold = threshold

    return best_ams, best_threshold

# Get the dissertation directory
project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the processed data path
data_path = os.path.join(project_directory, "higgs", "results", "processed_data.pt")

data = torch.load(
    data_path,
    map_location="cpu",
    weights_only=False
)

print("Data loaded successfully!")
print("Training features:", data["X_train"].shape)
print("Training labels:", data["y_train"].shape)
print("Validation features:", data["X_validation"].shape)
print("Test features:", data["X_test"].shape)

# Load validation event weights and labels
w_validation = data["w_validation"].double()
y_validation = data["y_validation"]

# Identify signal and background events
signal_mask = y_validation == 1
background_mask = y_validation == 0

# Calculate the weight scaling factors
signal_factor = (
    data["total_signal_weight"] / w_validation[signal_mask].sum().item()
)

background_factor = (
    data["total_background_weight"] / w_validation[background_mask].sum().item()
)

# Renormalise validation event weights
w_validation_ams = w_validation.clone()

w_validation_ams[signal_mask] *= signal_factor
w_validation_ams[background_mask] *= background_factor

print("\nValidation weight scaling factors:")
print("Signal:", signal_factor)
print("Background:", background_factor)

print("\nRenormalised validation weight totals:")
print("Signal:", w_validation_ams[signal_mask].sum().item())
print("Background:", w_validation_ams[background_mask].sum().item())

train_dataset = TensorDataset(
    data["X_train"],
    data["y_train"]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1024,
    shuffle=True
)

print("Number of training batches:", len(train_loader))

validation_dataset = TensorDataset(
    data["X_validation"],
    data["y_validation"]
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=1024,
    shuffle=False
)

print("Number of validation batches:", len(validation_loader))

model = HiggsClassifier()
model_config = get_model_config(model)

#Loss function
criterion = torch.nn.BCEWithLogitsLoss()

#adam intialization
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005
)

# save the best model based on validation AUC
best_val_ams = 0.0
best_threshold = 0.0

model_save_path = os.path.join(
    project_directory,
    "higgs",
    "results",
    "best_model.pth"
)

num_epochs = 20

training_config = get_training_config(
    model,
    optimizer,
    criterion,
    train_loader,
    validation_loader,
    num_epochs
)

start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()

    # Training
    model.train()
    total_train_loss = 0

    for features, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(features)
        labels = labels.unsqueeze(1)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * features.size(0)

    average_train_loss = total_train_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    total_validation_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for features, labels in validation_loader:

            outputs = model(features)
            labels = labels.unsqueeze(1)

            loss = criterion(outputs, labels)
            total_validation_loss += loss.item() * features.size(0)

            probabilities = torch.sigmoid(outputs)
            all_predictions.extend(probabilities.squeeze(1).tolist())
            all_labels.extend(labels.squeeze(1).tolist())

    average_validation_loss = total_validation_loss / len(validation_loader.dataset)

    #evaluation metrics
    roc_auc = roc_auc_score(all_labels, all_predictions)

    # Find the threshold that gives the highest validation AMS
    ams, threshold = find_best_ams_threshold(
        all_labels,
        all_predictions,
        w_validation_ams
    )

    # Save the model if it achieves a better validation AMS
    if ams > best_val_ams:

        best_val_ams = ams
        best_threshold = threshold

        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "training_config": training_config,
            "best_val_ams": best_val_ams,
            "best_threshold": best_threshold,
            "epoch": epoch + 1
        }, model_save_path)

        print("Best model saved!")

    predicted_classes = [1 if p >= 0.5 else 0 for p in all_predictions]

    f1 = f1_score(all_labels, predicted_classes)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Training Loss: {average_train_loss:.4f}, "
        f"Validation Loss: {average_validation_loss:.4f}, "
        f"ROC-AUC: {roc_auc:.4f}, "
        f"AMS: {ams:.4f}, "
        f"AMS Threshold: {threshold:.2f}, "
        f"F1-score: {f1:.4f}"
    )

    epoch_time = time.time() - epoch_start_time
    total_time = time.time() - start_time

    print(f"Epoch time: {epoch_time:.2f} seconds")
    print(f"Total elapsed time: {total_time / 60:.2f} minutes")

