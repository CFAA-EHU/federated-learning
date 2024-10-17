import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
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


# Cargar el dataset combinado
df = pd.read_csv("C:\\Users\\836582\\Downloads\\federated_working_powerZ_GMTK.csv")

# Definir las etiquetas personalizadas basadas en los rangos dados para la columna 'potenciaKW'
bins = [-float('inf'), 1, 4.118, float('inf')]
labels = ['Bajo', 'Medio', 'Alto']

# Crear la nueva columna de etiqueta con pd.cut usando los rangos personalizados
df['consumo_potencia'] = pd.cut(df['potenciaKW'], bins=bins, labels=labels)

# Graficar los puntos de 'load_Z' vs 'power_Z', coloreados por la etiqueta de consumo
plt.figure(figsize=(10, 6))

# Crear un scatter plot donde los colores representan las clases de consumo
colors = df['consumo_potencia'].map({'Bajo': 'green', 'Medio': 'yellow', 'Alto': 'red'})

plt.scatter(df['powerDrive_SPINDLE'], df['load_Z'], c=colors, label="Potencia", alpha=0.7)
plt.title("Distribución de powerDrive_SPINDLE vs load_Z por Etiqueta de Consumo")
plt.xlabel("powerDrive_SPINDLE")
plt.ylabel("load_Z")
plt.grid(True)
plt.show()