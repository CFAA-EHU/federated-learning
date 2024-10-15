import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cpu")  # Cambiar a "cuda" si tienes GPU disponible
print(f"Training on {DEVICE}")

BATCH_SIZE = 120
#EPOCHS = 60
EPOCHS = 30  # O 10 si prefieres


# 1. Cargar y preparar el dataset
def load_dataset():
    df = pd.read_csv("/home/ubuntu/aprendizaje_federado/merged_datasets.csv")  # Actualiza con tu ruta

    # Seleccionar características (X) y la variable objetivo (y)
    X = df[['load_X', 'load_Z', 'power_X', 'power_Z', 'load_SPINDLE', 'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE']].values
    y = df['potenciaKW'].values

    # Dividir en conjunto de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val

# 2. Convertir los datos a tensores y cargarlos en DataLoader
def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=120):
    # Normalizamos las características
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    # Convertimos los datos a tensores
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train_scaled, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val_scaled, dtype=torch.float32))

    # Creamos los DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, scaler_X, scaler_y

# 3. Definición de la red neuronal
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8, 16)  # 8 características de entrada
        self.fc3 = nn.Linear(16, 1)  # 1 valor de salida para regresión

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x

# 4. Función de entrenamiento
def train(net, trainloader, epochs: int, verbose=False):
    """Entrena la red neuronal en el conjunto de entrenamiento."""
    criterion = nn.MSELoss()  # Función de pérdida para regresión
    optimizer = torch.optim.Adam(net.parameters(), lr=0.009)
    net.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in trainloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))  # Ajuste para que las dimensiones coincidan
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(trainloader)
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}")

# 5. Función de validación/test
def test(net, valloader):
    """Evalúa la red neuronal en el conjunto de validación."""
    criterion = nn.MSELoss()
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in valloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))  # Ajuste para que las dimensiones coincidan
            val_loss += loss.item()

    val_loss /= len(valloader)
    return val_loss

# 6. Carga de los datos y entrenamiento centralizado
X_train, X_val, y_train, y_val = load_dataset()
trainloader, valloader, scaler_X, scaler_y = create_data_loaders(X_train, y_train, X_val, y_val)

net = Net().to(DEVICE)

# Entrena el modelo durante 5 épocas de forma centralizada
for epoch in range(EPOCHS):
    train(net, trainloader, 1, verbose=True)
    val_loss = test(net, valloader)
    print(f"Epoch {epoch+1}: validation loss {val_loss}")

# Guarda el modelo preentrenado
torch.save(net.state_dict(), "pretrained_regression_model_best_params.pth")
