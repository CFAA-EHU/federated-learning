import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
## IBARMIA

'''
df = pd.read_csv('C:\\Users\\836582\\Downloads\\federado_IBARMIA.csv')

# Las columnas que queremos mantener
selected_columns = [
    'LoadX_IBARMIA',
    'LoadZ_IBARMIA',
    'PowerX_IBARMIA',
    'PowerZ_IBARMIA',
    'SpindleDriveLoad_IBARMIA',
    'SpindleActualSpeed_IBARMIA',
    'SpindleOverride_IBARMIA',
    'SPAxisPowerDrive',
    'potenciaKW',
    'precioPorKW'
]


# Filtrar las columnas seleccionadas
filtered_df = df[selected_columns]

# Guardar el nuevo CSV con las columnas filtradas
filtered_df.to_csv('C:\\Users\\836582\\Downloads\\federado_IBARMIA.csv', index=False)
'''

'''
df = pd.read_csv('C:\\Users\\836582\\Downloads\\federado_IBARMIA.csv')

# Diccionario para reemplazar los nombres de las columnas
nuevos_nombres = {
    'LoadX_IBARMIA': 'load_X',
    'LoadZ_IBARMIA': 'load_Z',
    'PowerX_IBARMIA': 'power_X',
    'PowerZ_IBARMIA': 'power_Z',
    'SpindleDriveLoad_IBARMIA': 'load_SPINDLE',
    'SpindleActualSpeed_IBARMIA': 'speed_SPINDLE',
    'SpindleOverride_IBARMIA': 'override_SPINDLE',
    'SPAxisPowerDrive': 'powerDrive_SPINDLE',
    'potenciaKW': 'potenciaKW',
    'precioPorKW': 'precioPorKW'
}

# Reemplazar los nombres en el DataFrame
df.rename(columns=nuevos_nombres, inplace=True)

df.to_csv('C:\\Users\\836582\\Downloads\\federado_IBARMIA.csv', index=False)
'''



'''
## GMTK
df = pd.read_csv('C:\\Users\\836582\\Downloads\\federado_GMTK.csv')


# Las columnas que queremos mantener
selected_columns = [
    'aaLoadX1_GMTK',
    'aaLoadZ1_GMTK',
    'aaPowerX1_GMTK',
    'aaPowerZ1_GMTK',
    'driveLoadSpindle_GMTK',
    'actSpeedSpindle_GMTK',
    'speedOvrSpindle_GMTK',
    'C1AxisPowerDriveGMTK',
    'potenciaKW',
    'precioPorKW'
]


# Filtrar las columnas seleccionadas
filtered_df = df[selected_columns]

# Guardar el nuevo CSV con las columnas filtradas
filtered_df.to_csv('C:\\Users\\836582\\Downloads\\federado_GMTK.csv', index=False)
'''

'''
df = pd.read_csv('C:\\Users\\836582\\Downloads\\federado_GMTK.csv')


# Diccionario para reemplazar los nombres de las columnas
nuevos_nombres = {
    'aaLoadX1_GMTK': 'load_X',
    'aaLoadZ1_GMTK': 'load_Z',
    'aaPowerX1_GMTK': 'power_X',
    'aaPowerZ1_GMTK': 'power_Z',
    'driveLoadSpindle_GMTK': 'load_SPINDLE',
    'actSpeedSpindle_GMTK': 'speed_SPINDLE',
    'speedOvrSpindle_GMTK': 'override_SPINDLE',
    'C1AxisPowerDriveGMTK': 'powerDrive_SPINDLE',
    'potenciaKW': 'potenciaKW',
    'precioPorKW': 'precioPorKW'
}

# Reemplazar los nombres en el DataFrame
df.rename(columns=nuevos_nombres, inplace=True)

df.to_csv('C:\\Users\\836582\\Downloads\\federado_GMTK.csv', index=False)
'''


'''
## DANOBAT
df = pd.read_csv('C:\\Users\\836582\\Downloads\\federado_DANOBAT.csv')


# Las columnas que queremos mantener
selected_columns = [
    'X1AxisLoad_DANOBAT',
    'Z1AxisLoad_DANOBAT',
    'X1AxisPower_DANOBAT',
    'Z1AxisPower_DANOBAT',
    'SpindleLoad_DANOBAT',
    'SpindleSpeed_DANOBAT',
    'SpindleOverride_DANOBAT',
    'SpindlePower',
    'potenciaKW',
    'precioPorKW'
]


# Filtrar las columnas seleccionadas
filtered_df = df[selected_columns]

# Guardar el nuevo CSV con las columnas filtradas
filtered_df.to_csv('C:\\Users\\836582\\Downloads\\federado_DANOBAT.csv', index=False)
'''

'''
df = pd.read_csv('C:\\Users\\836582\\Downloads\\federado_DANOBAT.csv')



# Diccionario para reemplazar los nombres de las columnas
nuevos_nombres = {
    'X1AxisLoad_DANOBAT': 'load_X',
    'Z1AxisLoad_DANOBAT': 'load_Z',
    'X1AxisPower_DANOBAT': 'power_X',
    'Z1AxisPower_DANOBAT': 'power_Z',
    'SpindleLoad_DANOBAT': 'load_SPINDLE',
    'SpindleSpeed_DANOBAT': 'speed_SPINDLE',
    'SpindleOverride_DANOBAT': 'override_SPINDLE',
    'SpindlePower': 'powerDrive_SPINDLE',
    'potenciaKW': 'potenciaKW',
    'precioPorKW': 'precioPorKW'
}

# Reemplazar los nombres en el DataFrame
df.rename(columns=nuevos_nombres, inplace=True)

df.to_csv('C:\\Users\\836582\\Downloads\\federado_DANOBAT.csv', index=False)
'''


## media y desviacion tipica
'''
# Cargar el archivo CSV que generamos previamente
df = pd.read_csv('C:\\Users\\836582\\Downloads\\federado_DANOBAT.csv')

# Las columnas a analizar
columnas = ['load_X', 'load_Z', 'power_X', 'power_Z', 'load_SPINDLE', 'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE', 'potenciaKW', 'precioPorKW']

statistics = pd.DataFrame(columns=["variable", "media", "desviacion_tipica"])

# Calcular la media y desviación estándar para cada columna en df
for column in columnas:
    media = df[column].mean()
    desviacion_tipica = df[column].std()
    statistics = statistics._append({
        "variable": column,
        "media": media,
        "desviacion_tipica": desviacion_tipica
    }, ignore_index=True)



statistics.to_csv('C:\\Users\\836582\\Downloads\\cc.csv', index=False)
'''


'''
########### data profiling ############
# Leer los datos combinados
datos_combinados = pd.read_csv('C:\\Users\\836582\\Downloads\\federado_GMTK.csv')

# Generar el informe exploratorio
perfil = ProfileReport(datos_combinados, title='Informe Exploratorio de Datos', explorative=True)

# Guardar el informe en un archivo HTML
perfil.to_file('C:\\Users\\836582\\Downloads\\federado_GMTK_profiling.html')
########### data profiling ############
'''





#### QUITAR DE LOS DATASETS LOS DATOS QUE LA POTENCIA SEA 0 o MUY SIMILAR A 0

'''
df = pd.read_csv('C:\\Users\\836582\\Downloads\\federado_IBARMIA.csv')

df_filtered = df[~((df['power_X'] == 0) & (df['power_Z'] == 0))]

# Si deseas sobreescribir el dataset original
df = df_filtered

df.to_csv('C:\\Users\\836582\\Downloads\\federated_working_IBARMIA.csv', index=False)
print(len(df))
'''


'''
########### data profiling ############
# Leer los datos combinados
datos_combinados = pd.read_csv('C:\\Users\\836582\\Downloads\\federated_working_powerZ_IBARMIA.csv')

# Generar el informe exploratorio
perfil = ProfileReport(datos_combinados, title='Informe Exploratorio de Datos', explorative=True)

# Guardar el informe en un archivo HTML
perfil.to_file('C:\\Users\\836582\\Downloads\\federated_working_powerZ_IBARMIA_profiling.html')
########### data profiling ############
'''


'''
# Cargar el dataset
df = pd.read_csv("C:\\Users\\836582\\Downloads\\federated_working_powerZ_IBARMIA.csv")  # Actualiza con tu ruta

# Eliminar las columnas 'load_SPINDLE' y 'power_X'
#df = df.drop(columns=['load_SPINDLE', 'power_X'])

# Filtrar las filas donde 'power_Z' no sea igual a 0
df = df[df['speed_SPINDLE'] != 0]

df.to_csv('C:\\Users\\836582\\Downloads\\federated_working_powerZ_IBARMIA.csv', index=False)
'''




### CREAR EL NUEVO DATASET CON LA COLUMNA DE CONSUMO ALTO, MEDIO O BAJO
'''
# Cargar el dataset
df = pd.read_csv("C:\\Users\\836582\\Downloads\\federated_working_powerZ_GMTK.csv")  # Actualiza con tu ruta


# Crear la columna "coste" como precioporKw * potenciaKW en ambos datasets
df['coste'] = df['precioPorKW'] * df['potenciaKW']


df.to_csv('C:\\Users\\836582\\Downloads\\consumos_luz_GMTK.csv', index=False)
'''


'''
import matplotlib.pyplot as plt

# Cargar el dataset combinado con la columna 'coste'
df = pd.read_csv("C:\\Users\\836582\\Downloads\\consumos_luz_IBARMIA.csv")

# Graficar la columna 'coste'
plt.figure(figsize=(10, 6))
plt.plot(df['coste'], label="Coste", color='blue')
plt.title("Coste por fila en el dataset combinado")
plt.xlabel("Índice")
plt.ylabel("Coste")
plt.legend()
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt

# Cargar el CSV que contiene la columna "coste"
df = pd.read_csv("C:\\Users\\836582\\Downloads\\consumos_luz_IBARMIA.csv")

# Calcular la media y desviación estándar de la columna "coste"
mean_cost = df['coste'].mean()
std_cost = df['coste'].std()

print(mean_cost)
print(std_cost)

# Definir las etiquetas según los criterios
conditions = [
    (df['coste'] < abs(mean_cost - std_cost)),
    (df['coste'] >= mean_cost - std_cost) & (df['coste'] <= mean_cost + std_cost),
    (df['coste'] > mean_cost + std_cost)
]
labels = ['Consumo bajo', 'Consumo medio', 'Consumo alto']

# Crear la nueva columna de etiqueta
df['consumo'] = pd.cut(df['coste'], bins=[-float('inf'), mean_cost - std_cost, mean_cost + std_cost, float('inf')], labels=labels)

# Mostrar la distribución en un gráfico
plt.figure(figsize=(8, 6))
df['consumo'].value_counts().plot(kind='bar', color=['green', 'yellow', 'red'])
plt.title('Distribución de Consumo: Bajo, Medio y Alto')
plt.xlabel('Categoría de Consumo')
plt.ylabel('Número de muestras')
plt.xticks(rotation=0)
plt.show()


# Mostrar los puntos en un gráfico de dispersión según las etiquetas
plt.figure(figsize=(10, 6))

# Colores para cada categoría
colors = {'Consumo bajo': 'green', 'Consumo medio': 'yellow', 'Consumo alto': 'red'}

# Graficar puntos de cada categoría
for label in labels:
    subset = df[df['consumo'] == label]
    plt.scatter(subset.index, subset['coste'], label=label, color=colors[label], alpha=0.5)

plt.title('Distribución de Coste por Categoría de Consumo')
plt.xlabel('Índice de las Instancias')
plt.ylabel('Coste')
plt.legend(title="Categoría de Consumo")
plt.grid(True)
plt.show()
'''

'''


# Cargar el dataset combinado con la columna 'coste'
df = pd.read_csv("C:\\Users\\836582\\Downloads\\federated_working_powerZ_GMTK.csv")

# Definir las etiquetas personalizadas basadas en los rangos dados
bins = [-float('inf'), 1, 4.118, float('inf')]
labels = ['Bajo', 'Medio', 'Alto']

# Crear la nueva columna de etiqueta con pd.cut usando los rangos personalizados
df['consumo_potencia'] = pd.cut(df['potenciaKW'], bins=bins, labels=labels)

# Mostrar la nueva columna con las etiquetas
print(df[['potenciaKW', 'consumo_potencia']].head())

# Graficar los puntos de la columna 'potenciaKW' etiquetados por consumo
plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['potenciaKW'], c=df['consumo_potencia'].map({'Bajo': 'green', 'Medio': 'yellow', 'Alto': 'red'}), label="Potencia")
plt.title("Potencia por fila con etiquetas de consumo (Bajo, Medio, Alto)")
plt.xlabel("Índice")
plt.ylabel("Potencia (KW)")
plt.grid(True)
plt.show()
'''

'''
# Cargar el dataset combinado
#df = pd.read_csv("C:\\Users\\836582\\Downloads\\federated_working_powerZ_GMTK.csv")
df = pd.read_csv("C:\\Users\\836582\\Downloads\\balanced_labeled_GMTK.csv")

# Definir las etiquetas personalizadas basadas en los rangos dados para la columna 'potenciaKW'
bins = [-float('inf'), 1, 4.118, float('inf')]
labels = ['Bajo', 'Medio', 'Alto']

# Crear la nueva columna de etiqueta con pd.cut usando los rangos personalizados
df['consumo_potencia'] = pd.cut(df['potenciaKW'], bins=bins, labels=labels)

# Guardar el dataframe con la nueva columna en un nuevo CSV
#output_path = "C:\\Users\\836582\\Downloads\\labeled_GMTK.csv"
#df.to_csv(output_path, index=False)

# Graficar los puntos de 'load_Z' vs 'power_Z', coloreados por la etiqueta de consumo
plt.figure(figsize=(10, 6))

# Crear un scatter plot donde los colores representan las clases de consumo
colors = df['consumo_potencia'].map({'Bajo': 'green', 'Medio': 'yellow', 'Alto': 'red'})


#'load_X', 'load_Z', 'power_Z', 'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE

plt.scatter(df['powerDrive_SPINDLE'], df['load_Z'], c=colors, label="Potencia", alpha=0.7)
plt.title("Distribución de powerDrive_SPINDLE vs load_Z por Etiqueta de Consumo")
plt.xlabel("powerDrive_SPINDLE")
plt.ylabel("load_Z")
plt.grid(True)
plt.show()
'''


'''
#### BALANCEAR EL DATASET
df = pd.read_csv("C:\\Users\\836582\\Downloads\\unbalanced_labeled_GMTK.csv")

# Contar las instancias de cada clase en la columna 'consumo_potencia'
#class_counts = df['consumo_potencia'].value_counts()

# Mostrar los resultados
#print(class_counts)


from sklearn.utils import resample

# Cargar el dataset
#df = pd.read_csv("C:\\Users\\836582\\Downloads\\federated_working_powerZ_labeled.csv")

# Separar las clases
df_medio = df[df['consumo_potencia'] == 'Medio']
df_alto = df[df['consumo_potencia'] == 'Alto']
df_bajo = df[df['consumo_potencia'] == 'Bajo']

# Hacer undersampling para la clase 'Medio' a 3000 instancias
df_medio_downsampled = resample(df_medio, replace=False, n_samples=1380, random_state=42)

# Mantener todas las instancias de las clases 'Alto' y 'Bajo'
df_balanced = pd.concat([df_medio_downsampled, df_alto, df_bajo])

# Mezclar el dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Guardar el dataset balanceado
df_balanced.to_csv("C:\\Users\\836582\\Downloads\\balanced_labeled_GMTK.csv", index=False)

# Mostrar el nuevo conteo de instancias por clase
print(df_balanced['consumo_potencia'].value_counts())
'''



### Generame un csv con 1000 instancias de cada clase
### y otro csv con las que sobran para utilizarlo despues para el test
'''
# Cargar el dataset balanceado
df = pd.read_csv("C:\\Users\\836582\\Downloads\\balanced_labeled_GMTK.csv")

# Contar instancias por clase
print(df['consumo_potencia'].value_counts())

# Separar las instancias de cada clase
bajo = df[df['consumo_potencia'] == 'Bajo']
medio = df[df['consumo_potencia'] == 'Medio']
alto = df[df['consumo_potencia'] == 'Alto']

# Seleccionar 1000 instancias de cada clase (si hay suficientes)
bajo_sample = bajo.sample(n=1000, random_state=42) if len(bajo) >= 1000 else bajo
medio_sample = medio.sample(n=1000, random_state=42) if len(medio) >= 1000 else medio
alto_sample = alto.sample(n=1000, random_state=42) if len(alto) >= 1000 else alto

# Crear un DataFrame con las muestras seleccionadas
selected_samples = pd.concat([bajo_sample, medio_sample, alto_sample])

# Guardar el DataFrame con 1000 instancias de cada clase
selected_samples.to_csv("C:\\Users\\836582\\Downloads\\selected_samples.csv", index=False)

# Crear un DataFrame con las instancias restantes
remaining_samples = df[~df.index.isin(selected_samples.index)]

# Guardar el DataFrame con las instancias restantes
remaining_samples.to_csv("C:\\Users\\836582\\Downloads\\remaining_samples.csv", index=False)

print(f"Se han guardado {len(selected_samples)} instancias en 'selected_samples.csv'.")
print(f"Se han guardado {len(remaining_samples)} instancias en 'remaining_samples.csv'.")
'''




### ENTRENAMIENTO DE LA RED NEURONAL
'''
# 1. Cargar y preparar el dataset
df = pd.read_csv("C:\\Users\\836582\\Downloads\\balanced_labeled_GMTK.csv")

# Seleccionar características (X) y la variable objetivo (y)
X = df[['load_X', 'load_Z', 'power_Z', 'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE']].values
y = df['consumo_potencia'].values

# Codificar las etiquetas (Bajo, Medio, Alto) a valores numéricos
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir en conjunto de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

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
        self.fc2 = nn.Linear(16, 3)  # Capa de salida con 3 clases (bajo, medio, alto)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # No aplicamos softmax aquí porque usamos CrossEntropyLoss
        return x


# 3. Entrenamiento de la red neuronal
def train_model(net, train_loader, val_loader, epochs=30, lr=0.001):
    criterion = nn.CrossEntropyLoss()  # Función de pérdida para clasificación multiclase
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Evaluación en el conjunto de validación
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = net(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                # Predecir las clases con la mayor probabilidad
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_loss /= len(val_loader)
        accuracy = correct / total

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}")


# 4. Ejecutar el modelo
net = Net()
train_model(net, train_loader, val_loader, epochs=30)
'''


## matriz de confusion

'''
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

# Cargar el dataset balanceado
df = pd.read_csv("C:\\Users\\836582\\Downloads\\balanced_labeled_GMTK.csv")

# Separar características (X) y la variable objetivo (y)
X = df[['load_X', 'load_Z', 'power_Z', 'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE']].values
y = df['consumo_potencia'].map({'Bajo': 0, 'Medio': 1, 'Alto': 2}).values  # Mapeo de las clases a valores numéricos

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir el modelo neuronal (como lo mencionaste anteriormente)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(6, 16)  # 6 características de entrada, 16 neuronas en la capa oculta
        self.fc2 = nn.Linear(16, 3)  # 3 clases de salida (Bajo, Medio, Alto)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # No necesitamos softmax ya que usaremos CrossEntropyLoss que lo incluye
        return x

# Convertir los datos escalados a tensores
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Para clasificación, etiquetas deben ser tipo long
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Crear el modelo
model = SimpleNN()

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenar el modelo
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Hacer predicciones en el conjunto de prueba
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, y_pred = torch.max(test_outputs, 1)  # Obtener las clases predichas


# Calcular la precisión (accuracy)
accuracy = accuracy_score(y_test_tensor, y_pred)
print(f'Precisión (Accuracy): {accuracy:.4f}')

# Generar el informe de clasificación con métricas detalladas
report = classification_report(y_test_tensor, y_pred, target_names=['Bajo', 'Medio', 'Alto'])
print("\nInforme de clasificación:")
print(report)

# Generar la matriz de confusión
cm = confusion_matrix(y_test_tensor, y_pred)

# Mostrar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bajo', 'Medio', 'Alto'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.show()
'''


from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Cargar y preparar el dataset de entrenamiento
df_train = pd.read_csv("C:\\Users\\836582\\Downloads\\balanced_labeled_GMTK.csv")

# Seleccionar características (X) y la variable objetivo (y)
X = df_train[['load_X', 'load_Z', 'power_Z', 'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE']].values
y = df_train['consumo_potencia'].values

# Codificar las etiquetas (Bajo, Medio, Alto) a valores numéricos
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir en conjunto de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

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

# 3. Entrenamiento de la red neuronal
def train_model(net, train_loader, val_loader, epochs=30, lr=0.001):
    criterion = nn.CrossEntropyLoss()  # Función de pérdida para clasificación multiclase
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Evaluación en el conjunto de validación
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = net(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                # Predecir las clases con la mayor probabilidad
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_loss /= len(val_loader)
        accuracy = correct / total

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}")

# 4. Ejecutar el modelo
net = Net()
train_model(net, train_loader, val_loader, epochs=100)

# 5. Cargar y preparar el dataset de prueba
df_test = pd.read_csv("C:\\Users\\836582\\Downloads\\remaining_samples.csv")

# Seleccionar características (X_test) y la variable objetivo (y_test)
X_test = df_test[['load_X', 'load_Z', 'power_Z', 'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE']].values
y_test = df_test['consumo_potencia'].values

# Codificar las etiquetas para el conjunto de prueba
y_test_encoded = label_encoder.transform(y_test)

# Normalizar los datos de entrada del conjunto de prueba
X_test_scaled = scaler.transform(X_test)

# Convertir los datos a tensores
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

# 6. Evaluar el modelo en el conjunto de prueba
def evaluate_model(net, X_test_tensor, y_test_tensor):
    net.eval()
    with torch.no_grad():
        outputs = net(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
        cm = confusion_matrix(y_test_tensor.numpy(), predicted.numpy())
    return accuracy, cm

# Calcular la precisión y la matriz de confusión
accuracy, cm = evaluate_model(net, X_test_tensor, y_test_tensor)
print(f"Test Accuracy: {accuracy:.4f}")


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bajo', 'Medio', 'Alto'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.show()

#print("Confusion Matrix:")
#print(cm)
