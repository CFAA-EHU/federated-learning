import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

# ======================
# Reproducibility
# ======================
#torch.manual_seed(42)
np.random.seed(42)

# ======================
# Load training data (6000 aggregated)
# ======================
df_train = pd.read_csv("/home/ubuntu/aprendizaje_federado/totalData.csv")

X = df_train[['load_X', 'load_Z', 'power_Z', 'speed_SPINDLE',
              'override_SPINDLE', 'powerDrive_SPINDLE']].values
y = df_train['consumo_potencia'].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                          batch_size=32, shuffle=True)

val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor),
                        batch_size=32)

# ======================
# Model (same as FL)
# ======================
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

# ======================
# Training
# ======================
def train(net, trainloader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in trainloader:
            optimizer.zero_grad()
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(trainloader)
        print(f"Epoch {epoch+1}/100 - Train Loss: {epoch_loss:.4f}")

# ======================
# Evaluation
# ======================
def evaluate(net, dataloader):
    net.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = net(X_batch)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(y_batch.tolist())
            y_pred.extend(predicted.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, f1

# ======================
# Train model (100 epochs)
# ======================
DEVICE = torch.device("cpu")
net = Net().to(DEVICE)

train(net, train_loader, epochs=100)

# ======================
# Load TEST SET (global)
# ======================
df_test = pd.read_csv("/home/ubuntu/aprendizaje_federado/test.csv")

X_test = df_test[['load_X', 'load_Z', 'power_Z', 'speed_SPINDLE',
                  'override_SPINDLE', 'powerDrive_SPINDLE']].values
y_test = df_test['consumo_potencia'].values

X_test_scaled = scaler.transform(X_test)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor),
                         batch_size=32)

# ======================
# Final evaluation
# ======================
acc, f1 = evaluate(net, test_loader)

print("\n=== CENTRALIZED MLP RESULTS ===")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")

# ======================
# Save model
# ======================
torch.save(net.state_dict(),
           "/home/ubuntu/aprendizaje_federado/centralized_model.pth")
