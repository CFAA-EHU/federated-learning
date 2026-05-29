from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


# =====================================================
# Configuration
# =====================================================
SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device("cpu")

FEATURE_COLUMNS = [
    "load_X",
    "load_Z",
    "power_Z",
    "speed_SPINDLE",
    "override_SPINDLE",
    "powerDrive_SPINDLE",
]

TARGET_COLUMN = "consumo_potencia"


# =====================================================
# Paths
# =====================================================
CLIENT_FILES = [
    Path("/home/ubuntu/aprendizaje_federado/ibarmia_alto.csv"),
    Path("/home/ubuntu/aprendizaje_federado/ibarmia_bajo.csv"),
    Path("/home/ubuntu/aprendizaje_federado/ibarmia_med1.csv"),
    Path("/home/ubuntu/aprendizaje_federado/ibarmia_med2.csv"),
]

GLOBAL_TEST_FILE = Path("/home/ubuntu/aprendizaje_federado/localOnlyMLP/mlp6000.csv")

OUTPUT_RESULTS = Path("local_only_results.csv")


# =====================================================
# Reproducibility
# =====================================================
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================================================
# Model: same architecture as federated clients
# =====================================================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# =====================================================
# Data utilities
# =====================================================
def check_columns(df: pd.DataFrame, file_path: Path):
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in {file_path}: {missing}")


def create_loaders_for_client(
    client_file: Path,
    global_test_file: Path,
    batch_size: int = 32,
):
    df_client = pd.read_csv(client_file)
    df_test = pd.read_csv(global_test_file)

    check_columns(df_client, client_file)
    check_columns(df_test, global_test_file)

    X_client = df_client[FEATURE_COLUMNS].values
    y_client = df_client[TARGET_COLUMN].values

    X_test = df_test[FEATURE_COLUMNS].values
    y_test = df_test[TARGET_COLUMN].values

    # Same 80/20 split as in federated client code
    X_train, X_val, y_train, y_val = train_test_split(
        X_client,
        y_client,
        test_size=0.2,
        random_state=SEED,
        stratify=y_client if len(np.unique(y_client)) > 1 else None,
    )

    # Fit scaler only on the local client training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# =====================================================
# Training and evaluation
# =====================================================
def train_local_model(net, train_loader, epochs: int):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    net.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        print(f"Epoch {epoch + 1:03d}/{epochs} - Train loss: {epoch_loss:.4f}")


def evaluate_model(net, data_loader):
    criterion = nn.CrossEntropyLoss()

    net.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = net(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    return avg_loss, accuracy, f1


# =====================================================
# Local-only baseline
# =====================================================
def run_local_only_baseline():
    set_seed(SEED)

    results = []

    for client_idx, client_file in enumerate(CLIENT_FILES, start=1):
        print("\n" + "=" * 70)
        print(f"Training local-only model for Client {client_idx}")
        print(f"Dataset: {client_file}")
        print("=" * 70)

        train_loader, val_loader, test_loader = create_loaders_for_client(
            client_file=client_file,
            global_test_file=GLOBAL_TEST_FILE,
            batch_size=BATCH_SIZE,
        )

        net = Net().to(DEVICE)

        train_local_model(
            net=net,
            train_loader=train_loader,
            epochs=EPOCHS,
        )

        val_loss, val_acc, val_f1 = evaluate_model(net, val_loader)
        test_loss, test_acc, test_f1 = evaluate_model(net, test_loader)

        print(f"\nClient {client_idx} validation:")
        print(f"  Loss:     {val_loss:.4f}")
        print(f"  Accuracy: {val_acc * 100:.2f}%")
        print(f"  F-score:  {val_f1 * 100:.2f}%")

        print(f"\nClient {client_idx} global test:")
        print(f"  Loss:     {test_loss:.4f}")
        print(f"  Accuracy: {test_acc * 100:.2f}%")
        print(f"  F-score:  {test_f1 * 100:.2f}%")

        results.append({
            "client": client_idx,
            "client_file": str(client_file),
            "val_loss": val_loss,
            "val_accuracy_percent": val_acc * 100,
            "val_fscore_percent": val_f1 * 100,
            "global_test_loss": test_loss,
            "global_test_accuracy_percent": test_acc * 100,
            "global_test_fscore_percent": test_f1 * 100,
        })

    results_df = pd.DataFrame(results)

    mean_acc = results_df["global_test_accuracy_percent"].mean()
    std_acc = results_df["global_test_accuracy_percent"].std(ddof=1)

    mean_f1 = results_df["global_test_fscore_percent"].mean()
    std_f1 = results_df["global_test_fscore_percent"].std(ddof=1)

    print("\n" + "=" * 70)
    print("LOCAL-ONLY BASELINE SUMMARY")
    print("=" * 70)
    print(results_df)
    print()
    print(f"Local-only MLP Accuracy: {mean_acc:.2f} ± {std_acc:.2f}")
    print(f"Local-only MLP F-score:  {mean_f1:.2f} ± {std_f1:.2f}")

    results_df.to_csv(OUTPUT_RESULTS, index=False)
    print(f"\nResults saved to: {OUTPUT_RESULTS}")


if __name__ == "__main__":
    run_local_only_baseline()
