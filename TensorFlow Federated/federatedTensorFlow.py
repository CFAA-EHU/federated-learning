import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

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
def aggregate_weights(weights, num_samples):
    total_samples = np.sum(num_samples)
    new_weights = []

    for weights_list in zip(*weights):
        weighted_sum = np.zeros_like(weights_list[0])
        for w, n in zip(weights_list, num_samples):
            weighted_sum += (n / total_samples) * w
        new_weights.append(weighted_sum)

    return new_weights

# Cargar y preparar datos para un cliente
def load_data(client_id):
    if client_id == 0:
        df = pd.read_csv("/home/ubuntu/tensorflow/ibarmia_alto.csv")
    elif client_id == 1:
        df = pd.read_csv("/home/ubuntu/tensorflow/ibarmia_bajo.csv")
    elif client_id == 2:
        df = pd.read_csv("/home/ubuntu/tensorflow/ibarmia_med1.csv")  # Cambia a tu archivo
    else:
        df = pd.read_csv("/home/ubuntu/tensorflow/ibarmia_med2.csv")  # Cambia a tu archivo

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
    num_samples = []
    round_loss, round_accuracy = [], []

    for client_id in range(NUM_CLIENTS):
        # Cargar datos para el cliente
        (X_train, y_train), _ = load_data(client_id)
        client_model = clients_models[client_id]

        # Entrenar el modelo del cliente
        #client_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        # Entrenar el modelo del cliente y calcular el train loss
        history = client_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        train_loss = history.history['loss'][0]  # Obtén el valor de la loss por ronda
        round_loss.append(train_loss)

        # Obtener pesos del modelo entrenado
        num_samples.append(len(X_train))
        weights.append(client_model.get_weights())

    # Agregar pesos de todos los modelos de los clientes
    aggregated_weights = aggregate_weights(weights, num_samples)

    # Actualizar todos los modelos de los clientes con los pesos agregados
    for client_model in clients_models:
        client_model.set_weights(aggregated_weights)

    # Evaluar el modelo después de cada ronda
    total_loss, total_accuracy, total_f1 = 0, 0, 0
    for client_id in range(NUM_CLIENTS):
        _, (X_val, y_val) = load_data(client_id)
        loss, accuracy = client_model.evaluate(X_val, y_val, verbose=0)
        total_loss += loss
        total_accuracy += accuracy

        # Calcular F1-Score para la validación
        y_val_pred = client_model.predict(X_val, batch_size=BATCH_SIZE)
        y_val_pred_classes = np.argmax(y_val_pred, axis=1)
        total_f1 += f1_score(y_val, y_val_pred_classes, average='weighted')

    # Promediar las métricas para esta ronda
    average_loss = total_loss / NUM_CLIENTS
    average_accuracy = total_accuracy / NUM_CLIENTS
    average_f1 = total_f1 / NUM_CLIENTS
    average_train_loss = np.mean(round_loss)  # Promedio de train loss por ronda
    print(f"Round {round_num + 1}: Val Loss: {average_loss:.6f}, Accuracy: {average_accuracy:.6f}, Train Loss: {average_train_loss:.6f}, Fscore: {average_f1:.6f}")
    #print(f"Round {round_num + 1}: Train Loss: {average_train_loss:.6f}, Fscore: {average_f1:.6f}")
