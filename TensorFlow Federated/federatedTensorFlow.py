import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuración
NUM_CLIENTS = 4  # Número de clientes
NUM_ROUNDS = 100  # Número de rondas de entrenamiento
BATCH_SIZE = 32
EPOCHS = 1

# Definición del modelo
def create_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(6,)),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')  # Salida para 3 clases
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Función para agregar pesos de los modelos
def aggregate_weights(weights):
    new_weights = []
    for weights_list in zip(*weights):
        new_weights.append(np.mean(weights_list, axis=0))
    return new_weights

# Cargar y preparar datos para un cliente
def load_data(client_id):
    if client_id == 0:
        df = pd.read_csv("/home/ubuntu/tensorflow/alto.csv")
    elif client_id == 1:
        df = pd.read_csv("/home/ubuntu/tensorflow/bajo.csv")
    elif client_id == 2:
        df = pd.read_csv("/home/ubuntu/tensorflow/med1.csv")  # Cambia a tu archivo
    else:
        df = pd.read_csv("/home/ubuntu/tensorflow/med2.csv")  # Cambia a tu archivo

    X = df[['load_X', 'load_Z', 'power_Z', 'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE']].values
    y = df['consumo_potencia'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # Normalización
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    return (X_train, y_train), (X_val, y_val)

# Simulación de clientes
clients_models = [create_model() for _ in range(NUM_CLIENTS)]


# Entrenamiento del servidor
for round_num in range(NUM_ROUNDS):
    print(f"Round {round_num + 1}/{NUM_ROUNDS}")
    weights = []
    round_loss, round_accuracy = [], []

    for client_id in range(NUM_CLIENTS):
        # Cargar datos para el cliente
        (X_train, y_train), _ = load_data(client_id)
        client_model = clients_models[client_id]

        # Entrenar el modelo del cliente
        client_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        # Obtener pesos del modelo entrenado
        weights.append(client_model.get_weights())

    # Agregar pesos de todos los modelos de los clientes
    aggregated_weights = aggregate_weights(weights)

    # Actualizar todos los modelos de los clientes con los pesos agregados
    for client_model in clients_models:
        client_model.set_weights(aggregated_weights)

    # Evaluar el modelo después de cada ronda
    total_loss, total_accuracy = 0, 0
    for client_id in range(NUM_CLIENTS):
        _, (X_val, y_val) = load_data(client_id)
        loss, accuracy = client_model.evaluate(X_val, y_val, verbose=0)
        total_loss += loss
        total_accuracy += accuracy
    # Promediar las métricas para esta ronda
    average_loss = total_loss / NUM_CLIENTS
    average_accuracy = total_accuracy / NUM_CLIENTS
    print(f"Round {round_num + 1}: Loss: {average_loss:.4f}, Accuracy: {average_accuracy:.4f}")









'''
# Evaluar el modelo final en el conjunto de validación
final_model = create_model()  # Modelo para la evaluación
# Cargar datos de validación del primer cliente (puedes ajustar esto según tus necesidades)
_, (X_val, y_val) = load_data(0)  # Aquí evaluamos solo con datos del cliente 0

# Aplicar pesos agregados
final_model.set_weights(aggregated_weights)  # Aplicar pesos agregados
loss, accuracy = final_model.evaluate(X_val, y_val, verbose=0)

print(f"Final validation loss: {loss:.4f}, accuracy: {accuracy:.4f}")
'''

'''
# Evaluar el modelo final en el conjunto de validación
final_model = create_model()  # Modelo para la evaluación
# Puedes evaluar con datos de uno de los clientes o promediar sobre todos
val_data = []
for client_id in range(NUM_CLIENTS):
    _, (X_val, y_val) = load_data(client_id)
    val_data.append((X_val, y_val))

# Evaluar el modelo final en todos los datos de validación
total_loss, total_accuracy = 0, 0
for X_val, y_val in val_data:
    final_model.set_weights(aggregated_weights)  # Aplicar pesos agregados
    loss, accuracy = final_model.evaluate(X_val, y_val, verbose=0)
    total_loss += loss
    total_accuracy += accuracy

# Promediar las métricas
average_loss = total_loss / NUM_CLIENTS
average_accuracy = total_accuracy / NUM_CLIENTS

print(f"Final validation loss: {average_loss:.4f}, accuracy: {average_accuracy:.4f}")
'''
