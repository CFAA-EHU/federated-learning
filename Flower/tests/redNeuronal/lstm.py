import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Cargar el dataset
data = pd.read_csv('C:\\Users\\836582\\Downloads\\consumoPorHora_IBARMIA.csv')

# Convertir timestamps a datetime (opcional)
data['timestamp_inicio'] = pd.to_datetime(data['timestamp_inicio'], unit='ns')
data['timestamp_fin'] = pd.to_datetime(data['timestamp_fin'], unit='ns')

# Seleccionar las columnas de interés para el modelo (consumo y precio)
#dataset = data[['consumo_total_kwh', 'precio_luz']].values
dataset = data[['consumo_total_kwh']].values

# Normalizar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

# Crear secuencias de tiempo
def crear_secuencias(data, ventana):
    X, y = [], []
    for i in range(len(data) - ventana):
        X.append(data[i:i + ventana, :])
        y.append(data[i + ventana, 0])  # Predecir solo el consumo (consumo_total_kwh)
    return np.array(X), np.array(y)

ventana_tiempo = 24  # Definir la ventana de tiempo

# Crear las secuencias de entrada y salida
X, y = crear_secuencias(dataset_scaled, ventana_tiempo)

# Convertir a tensores de PyTorch
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Dividir los datos en entrenamiento y prueba
train_size = int(0.8 * len(X))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Capa de salida para predecir el consumo total (1 valor)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Propagación hacia adelante a través de la LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Solo la última salida
        out = self.fc(out[:, -1, :])
        return out


# Parámetros del modelo
#input_size = X_train.shape[2]  # Número de características de entrada (consumo y precio)
input_size = 1
hidden_size = 50
num_layers = 2

# Instanciar el modelo
model = LSTMModel(input_size, hidden_size, num_layers)

# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()  # Error cuadrático medio
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Mover el modelo a GPU si está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Entrenamiento
num_epochs = 50
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size].to(device)
        y_batch = y_train[i:i + batch_size].to(device)

        # Hacer una predicción
        outputs = model(X_batch)

        # Calcular la pérdida
        loss = criterion(outputs, y_batch.unsqueeze(1))

        # Backpropagation y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(X_train):.4f}')



model.eval()
with torch.no_grad():
    # Hacer predicciones sobre el conjunto de prueba
    y_pred = model(X_test.to(device))
    y_pred = y_pred.cpu().numpy()

# Invertir la normalización para obtener los valores reales
#y_test_inv = scaler.inverse_transform(np.concatenate((y_test.unsqueeze(1).numpy(), X_test[:, -1, 1].unsqueeze(1).numpy()), axis=1))
y_test_inv = scaler.inverse_transform(np.concatenate((y_test.unsqueeze(1).numpy(), X_test[:, -1, 0].unsqueeze(1).numpy()), axis=1))

#y_pred_inv = scaler.inverse_transform(np.concatenate((y_pred, X_test[:, -1, 1].unsqueeze(1).cpu().numpy()), axis=1))
y_pred_inv = scaler.inverse_transform(np.concatenate((y_pred, X_test[:, -1, 0].unsqueeze(1).cpu().numpy()), axis=1))

# Graficar los resultados
plt.plot(y_test_inv[:, 0], color='blue', label='Consumo Real (kWh)')
plt.plot(y_pred_inv[:, 0], color='red', label='Consumo Predicho (kWh)')
plt.title('Predicción de Consumo usando LSTM en PyTorch')
plt.xlabel('Tiempo')
plt.ylabel('Consumo (kWh)')
plt.legend()
plt.show()
