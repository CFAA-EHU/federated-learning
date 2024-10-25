import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Cargar y preparar el dataset de entrenamiento
df_train = pd.read_csv("/home/ubuntu/aprendizaje_federado/ss.csv")

# Seleccionar características (X) y la variable objetivo (y)
X = df_train[['load_X', 'load_Z', 'power_Z', 'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE']].values
y = df_train['consumo_potencia'].values

# Dividir en conjunto de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos de entrada
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Convertir los datos a tensores
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Crear datasets de entrenamiento y validación
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

# Crear DataLoader para dividir los datos en batches
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 2. Definir la red neuronal
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 16)  # Capa de entrada con 6 características, y capa oculta con 16 neuronas
        self.fc2 = nn.Linear(16, 3)   # Capa de salida con 3 clases (bajo, medio, alto)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # No aplicamos softmax aquí porque usamos CrossEntropyLoss
        return x


def train(net, trainloader, epochs: int, verbose=False):
    criterion = nn.CrossEntropyLoss()  # Función de pérdida para clasificación multiclase
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in trainloader:  # Iterar sobre el DataLoader
            optimizer.zero_grad()  # Resetear gradientes
            outputs = net(X_batch)  # Hacer predicciones
            loss = criterion(outputs, y_batch)  # Calcular la pérdida
            loss.backward()  # Retropropagar el error
            optimizer.step()  # Actualizar los pesos

            # Acumular la pérdida y las métricas
            epoch_loss += loss.item()  # Acumular la pérdida de este batch
            #total += y_batch.size(0)  # Total de ejemplos en este batch
            #_, predicted = torch.max(outputs, 1)  # Obtener las predicciones
            #correct += (predicted == y_batch).sum().item()  # Comparar con etiquetas reales
            #correct += (torch.max(outputs.data, 1)[1] == y_batch).sum().item()

        # Calcular métricas promedio para la época actual
        epoch_loss /= len(trainloader)  # Promedio de la pérdida
        #epoch_acc = correct / total  # Precisión como proporción de aciertos

        if verbose:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}")


def test(net, testloader):
    criterion = nn.CrossEntropyLoss()  # Función de pérdida
    val_loss = 0.0
    correct = 0
    total = 0
    net.eval()

    with torch.no_grad():  # Desactivar el cálculo de gradientes para ahorrar memoria y acelerar
        for X_batch, y_batch in testloader:  # Iterar sobre el DataLoader de prueba
            outputs = net(X_batch)  # Hacer predicciones
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            # Predecir las clases con la mayor probabilidad
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    val_loss /= len(testloader)
    accuracy = correct / total
    return val_loss, accuracy


DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
net = Net().to(DEVICE)

# Entrena el modelo durante 5 épocas de forma centralizada
for epoch in range(5):
    train(net, train_loader, 1, verbose=True)
    #loss, accuracy = test(net, val_loader)
    loss, accuracy = test(net, val_loader)
    print(f"Epoch {epoch+1}: validation loss {loss}, validation accuracy {accuracy}")


# 5. Cargar y preparar el dataset de prueba
df_test = pd.read_csv("/home/ubuntu/aprendizaje_federado/rs.csv")

# Seleccionar características (X_test) y la variable objetivo (y_test)
X_test = df_test[['load_X', 'load_Z', 'power_Z', 'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE']].values
y_test = df_test['consumo_potencia'].values

# Normalizar los datos de entrada del conjunto de prueba
X_test_scaled = scaler.transform(X_test)

# Convertir los datos a tensores
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)



# Crear el dataset de prueba
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Crear DataLoader para dividir los datos en batches de tamaño 32
test_loader = DataLoader(test_dataset, batch_size=32)


# Evalúa en el conjunto de test después de entrenar centralizadamente
loss, accuracy = test(net, test_loader)
print(f"Final test set performance:\n\tloss {loss}\n\t Test accuracy {accuracy}")

# Guarda el modelo preentrenado
torch.save(net.state_dict(), "/home/ubuntu/aprendizaje_federado/pretrained_model_bnn.pth")
