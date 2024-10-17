from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
disable_progress_bar()


################### LOAD DATASETS AND CREATE TRAIN/VAL SETS FOR 10 CLIENTS ################
NUM_CLIENTS = 10 # sistema de aprendizaje federado para simular 10 clientes.
BATCH_SIZE = 32 #Los datos se cargarán en lotes de 32 imágenes cada vez que entrenes el modelo.

# Carga el dataset CIFAR-10 y lo particiona en 10 partes usando la clase FederatedDataset,
# una partición para cada cliente. Esto simula que cada cliente tiene una parte de los datos.
def load_datasets(partition_id: int):
    # Estás simulando que tienes varios clientes (en tu caso, 10).
    # Cada cliente recibe una porción del conjunto de datos
    # CIFAR-10 contiene 60,000 imágenes a color de 32x32 píxeles.
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    # Separa los datos en conjuntos de entrenamiento (80% - 50,000) y validación (20% - 10,000)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Aplica las transformaciones necesarias a las imágenes, como la normalización.
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Esto simplemente aplica las transformaciones a cada imagen en el lote
    # para que estén listas para ser procesadas por el modelo.
    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Crea los DataLoader para los conjuntos de entrenamiento y validación.
    # Estos son objetos que ayudan a cargar las imágenes en lotes para que
    # el modelo pueda procesarlas en partes, y no todas de golpe
    partition_train_test = partition_train_test.with_transform(apply_transforms)

    # Este es el "cargador" del conjunto de entrenamiento. Se encargará de pasar
    # las imágenes del conjunto de entrenamiento (80%) al modelo en lotes de tamaño 32
    trainloader = DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True)

    # Este es el "cargador" del conjunto de validación (20%).
    # Su función es comprobar cómo está aprendiendo el modelo.
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)

    # Este cargador maneja el conjunto de pruebas (test set), que contiene imágenes
    # que el modelo nunca ha visto. Después de entrenar, puedes usar estas imágenes
    # para probar si el modelo funciona bien con datos completamente nuevos
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader

    #trainloader: Este es el conjunto de datos principal que usa el modelo para aprender.
    # Son las imágenes que el modelo verá durante el entrenamiento y usará para ajustar sus "pesos"
    # (parámetros internos).

    #valloader: Este conjunto de datos se usa para verificar que el modelo esté aprendiendo
    # correctamente mientras entrena. Así puedes saber si el modelo está mejorando sin que lo memorice toda.

    #testloader: Después de entrenar al modelo, necesitas verificar si el modelo puede generalizar lo
    # aprendido a imágenes que nunca ha visto antes. El testloader te permite hacer eso.

################### LOAD DATASETS AND CREATE TRAIN/VAL SETS FOR 10 CLIENTS ################




##### Visualiza un lote de imágenes del conjunto de entrenamiento del cliente numero 0 ######

# Cargas los datos del cliente número 0 (la primera partición)
# y guardas el trainloader, que contiene las imágenes y etiquetas para entrenar.
trainloader, vall, testl = load_datasets(partition_id=0)

# Extraes un lote de imágenes y etiquetas del conjunto de entrenamiento (la partición 0 del cliente)
batch = next(iter(trainloader))
images, labels = batch["img"], batch["label"]

# Reshape and convert images to a NumPy array
# matplotlib requires images with the shape (height, width, 3)
images = images.permute(0, 2, 3, 1).numpy()

# Denormalize
images = images / 2 + 0.5

# Create a figure and a grid of subplots
fig, axs = plt.subplots(4, 8, figsize=(12, 6))

# Loop over the images and plot them
for i, ax in enumerate(axs.flat):
    ax.imshow(images[i])
    ax.set_title(trainloader.dataset.features["label"].int2str([labels[i]])[0])
    ax.axis("off")

# Show the plot
fig.tight_layout()
plt.show()
##### Visualiza un lote de imágenes del conjunto de entrenamiento del cliente numero 0 ######





########################### Step 1: Centralized Training with PyTorch #########################

# Ahora que ya tienes tus conjuntos de datos listos (entrenamiento, validación y pruebas),
# lo siguiente es definir el modelo de red neuronal y entrenarlo con esos datos.



# Definir una red neuronal que puede procesar las imágenes de CIFAR-10.
# Imagina que tu red neuronal es como una máquina que toma una imagen y pasa por varios "filtros" o "capas",
# hasta que al final te dice qué es lo que hay en la imagen (por ejemplo, si es un gato o un avión).

# 2 tipos de capas:  -Capas que detectan características en la imagen (bordes, formas, texturas).
#                    -Capas que toman esas características y hacen una decisión final sobre qué es la imagen.
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()

        # Capa 1: Detectar características básicas de la imagen
        self.conv1 = nn.Conv2d(3, 6, 5)  # Toma una imagen y detecta cosas sencillas

        # Reducir el tamaño de la imagen para enfocarse en lo importante
        self.pool = nn.MaxPool2d(2, 2)  # Reduce la imagen a algo más manejable

        # Capa 2: Detectar características más complejas
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Capas finales: Conectar toda lo que ha aprendido y decidir la clase (gato, avión, etc.)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # "Aplana" y procesa la imagen
        self.fc2 = nn.Linear(120, 84)  # Reduce los datos para tomar decisiones
        self.fc3 = nn.Linear(84, 10)  # Decidir entre 10 clases finales (gato, avión, etc.)

        # Capas convolucionales (conv1 y conv2): Son como "filtros" que detectan características
        # en las imágenes. Primero, detectan cosas simples como bordes, y luego cosas más complejas.

        # Pooling (pool): Después de detectar características, se reduce el tamaño de la imagen
        # para enfocarse solo en lo importante.

        # Capas completamente conectadas (fc1, fc2, fc3): Estas toman toda la información que los
        # filtros han detectado y deciden qué hay en la imagen (gato, avión, etc.).



    # Definir cómo la imagen pasa por la red
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # 1. Primer filtro + pooling
        # Después, aplicas ReLU a los resultados del filtro. ReLU se encarga de eliminar los valores negativos y
        # dejar pasar solo los positivos. Esto ayuda a la red a enfocarse en los aspectos importantes de la imagen.
        x = self.pool(F.relu(self.conv2(x)))  # 2. Segundo filtro + pooling
        x = x.view(-1, 16 * 5 * 5)  # 3. Aplanar la imagen
        x = F.relu(self.fc1(x))  # 4. Primera capa conectada
        x = F.relu(self.fc2(x))  # 5. Segunda capa conectada
        x = self.fc3(x)  # 6. Capa final que decide qué es la imagen
        return x

# Esta función tiene como objetivo entrenar la red neuronal utilizando el conjunto de datos de train (trainloader)
def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    # criterion (función de pérdida): La función de pérdida es la forma en que la red mide qué tan mal
    # están sus predicciones. En este caso, se utiliza CrossEntropyLoss, que es adecuada para tareas de
    # clasificación, donde tenemos varias clases (gato, perro, avión, etc.) y
    # queremos minimizar el error entre lo que predice la red y la etiqueta correcta.
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer (optimizador): El optimizador es quien ajusta los "pesos" de la red para mejorarla en cada paso
    # de entrenamiento. Aquí se usa el optimizador Adam, que es uno de los más eficientes para ajustar esos pesos.
    # El optimizador ajusta los parámetros de la red para que las predicciones sean más precisas.
    optimizer = torch.optim.Adam(net.parameters())
    net.train() # la red está en modo de entrenamiento

    # Bucle de entrenamiento por épocas
    # Aquí comienza el bucle de entrenamiento, donde repetimos el proceso de aprendizaje durante el número de épocas
    # que definimos. Una época es cuando la red ha pasado por todao el conjunto de datos una vez.
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0 # Inicializar métricas para la época
        # correct: Contará cuántas predicciones han sido correctas.
        # total: Contará cuántas imágenes hemos procesado en total.
        # epoch_loss: Acumulará el error (pérdida) de cada predicción a lo largo de la época.

        # Ahora, dentro de cada época, iteramos sobre los lotes de datos de entrenamiento
        #  Los datos se dividen en pequeños grupos llamados batches
        for batch in trainloader:
            # batch["img"] contiene las imágenes y batch["label"] contiene las etiquetas correspondientes
            # to(DEVICE): Mueve las imágenes y las etiquetas al dispositivo que estamos usando para entrenar (cpu)
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)

            # limpia los gradientes acumulados de los pasos anteriores. Los gradientes son valores que se calculan
            # para ajustar los "pesos" de la red, y antes de cada paso necesitamos reiniciar esos valores a cero.
            optimizer.zero_grad()

            # pasa las imágenes por la red neuronal para generar una predicción. El resultado es una lista
            # de probabilidades para cada clase (gato, perro, avión, etc.).
            outputs = net(images)

            # compara las predicciones de la red (outputs) con las etiquetas verdaderas (labels)
            # y calcula la pérdida (o error)
            loss = criterion(outputs, labels)

            # Calcula los gradientes, que indican cómo deben ajustarse los pesos de la red para reducir
            # la pérdida (o mejorar la predicción).
            loss.backward()
            # optimizer.step(): Ajusta los pesos de la red usando los gradientes calculados.
            # Este es el paso donde la red "aprende" y mejora su capacidad de predicción.
            optimizer.step()

            # Actualizar métricas
            epoch_loss += loss # Añade la pérdida del lote actual a la pérdida total de la época.
            total += labels.size(0) # Añade cuántas imágenes se han procesado en total.
            # Calcula cuántas predicciones han sido correctas en este lote y lo suma al contador de predicciones correctas.
            # Aquí usamos torch.max para obtener la clase con la probabilidad más alta.
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        # epoch_loss: Se calcula el promedio de la pérdida dividiendo la pérdida acumulada por el
        # número total de imágenes en el conjunto de entrenamiento.
        epoch_loss /= len(trainloader.dataset)
        # epoch_acc: La precisión se calcula dividiendo cuántas predicciones fueron
        # correctas por el total de imágenes procesadas.
        epoch_acc = correct / total

        #  Por ejemplo, si estás en la primera época, podrías ver algo como:
        # Epoch 1: train loss 0.005, accuracy 0.78
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")



# La función recibe la red ya entrenada (net) y el conjunto de datos de prueba (testloader).
# Su objetivo es:   -Medir el error (pérdida) de la red en el conjunto de test.
#                   -Calcular la precisión de la red (qué porcentaje de imágenes clasifica correctamente).
def test(net, testloader):
    """Evaluate the network on the entire test set."""
    # criterion (función de pérdida):  para medir qué tan mal están las predicciones de la red en el conjunto de test
    criterion = torch.nn.CrossEntropyLoss()

    correct, total, loss = 0, 0, 0.0 # Inicializar métricas
    #correct: Contará cuántas predicciones han sido correctas.
    #total: Contará cuántas imágenes del conjunto de prueba hemos procesado.
    #loss: Acumulará la pérdida total (error) de las predicciones en el conjunto de test.

    net.eval() # Cambia la red al modo de evaluación

    # Esto desactiva el cálculo de gradientes
    with torch.no_grad():
        # Aquí iteramos sobre los minilotes de datos de test (igual que lo hicimos en el entrenamiento).
        for batch in testloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)

            # Pasa las imágenes por la red entrenada y obtiene las predicciones
            # (probabilidades para cada clase, como si la imagen fuera un perro, un gato, etc.).
            outputs = net(images)

            # Compara las predicciones de la red (outputs) con las etiquetas correctas (labels)
            # y calcula el error (pérdida) usando la función de pérdida definida al principio.
            # item() --> Extrae el valor numérico del tensor para poder sumarlo a la pérdida total acumulada.
            loss += criterion(outputs, labels).item()

            # Aquí, de las probabilidades generadas por la red, seleccionamos la clase con la mayor probabilidad
            # como la predicción final de la red.
            # Ejemplo: Si la red predice [0.1, 0.8, 0.1], significa que la clase con más probabilidad
            # es la segunda (índice 1).


            _, predicted = torch.max(outputs.data, 1)
            # Sumamos cuántas imágenes hemos procesado en total.
            total += labels.size(0)

            # Aquí contamos cuántas predicciones han sido correctas comparando predicted con las etiquetas
            # verdaderas (labels). Si coinciden, significa que la red hizo una predicción correcta.
            correct += (predicted == labels).sum().item()

    # Calculamos la pérdida promedio dividiendo la pérdida total acumulada por el número total de
    # imágenes en el conjunto de prueba.
    loss /= len(testloader.dataset)
    # Calculamos la precisión dividiendo cuántas predicciones fueron correctas por el total de imágenes procesadas.
    accuracy = correct / total

    # Finalmente, la función retorna la pérdida y la precisión para todoa el conjunto de prueba
    return loss, accuracy






#Train the model
trainloader, valloader, testloader = load_datasets(partition_id=0)
#trainloader: Datos de entrenamiento (para ajustar los pesos del modelo).
#valloader: Datos de validación (para monitorear el rendimiento del modelo mientras se entrena).
#testloader: Datos de prueba (para evaluar el modelo al final del entrenamiento).

# Este modelo será el que entrenemos.
net = Net().to(DEVICE)

# Va a repetir el proceso de entrenamiento y validación durante 5 épocas.
# Una época significa que el modelo verá todas las imágenes del conjunto de entrenamiento una vez.
for epoch in range(5):
    # Llama a la función train, que entrena la red net usando el conjunto de datos trainloader.
    # El argumento 1 significa que en cada iteración del ciclo for, entrenamos el modelo
    # con todas las imágenes de entrenamiento una vez.
    train(net, trainloader, 1)

    # evaluamos cómo está funcionando el modelo en el conjunto de validación (valloader).
    # La función test calculará la pérdida y la precisión en el conjunto de validación,
    # pero sin modificar los pesos del modelo.
    loss, accuracy = test(net, valloader)

    # Después de cada época de entrenamiento, imprime la pérdida y precisión en el conjunto de validación
    print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")



# Evaluar el modelo en el conjunto de prueba al final
# Esta evaluación es clave porque el conjunto de prueba contiene imágenes que el modelo nunca ha visto
# (ni en el entrenamiento ni en la validación). Indican cómo de bien el modelo ha aprendido
# y qué tan bien generaliza a nuevos datos.
loss, accuracy = test(net, testloader)
print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
########################### Step 1: Centralized Training with PyTorch #########################








############################# Step 2: Federated Learning with Flower ########################


# Cargar los parámetros en el modelo
# Esta función actualiza los parámetros (pesos) de un modelo net en PyTorch usando una lista de matrices
# 'parameters' que se recibieron desde otro lugar (normalmente del servidor central en aprendizaje federado).
# Si el servidor central envía un conjunto de pesos actualizados para tu modelo,
# esta función se encarga de cargarlos en tu modelo local.
def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# Esta función extrae los parámetros actuales (pesos) del modelo net y los devuelve como una lista de arreglos
# de NumPy. Estos parámetros son los que se envían al servidor central en un entorno de aprendizaje federado.
# Al entrenar un modelo localmente en un cliente, esta función se usaría para obtener los pesos entrenados
# y enviarlos al servidor central.
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

# estas funciones permiten que el modelo local de cada cliente en el aprendizaje federado envíe y reciba
# los pesos del modelo global de manera eficiente,


# Los clientes son dispositivos que entrenan modelos localmente con sus propios datos. Cada cliente realiza
# el entrenamiento en su propio conjunto de datos y luego envía los parámetros (pesos) de su modelo
# al servidor central. El servidor central agrega (combina) esos parámetros para crear un modelo global.
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        # Es el modelo que se entrena en el cliente (en este caso, el modelo de red neuronal que definiste antes).
        self.net = net
        # El conjunto de datos de entrenamiento específico para este cliente.
        self.trainloader = trainloader
        # El conjunto de datos de validación específico para este cliente, que se usa para evaluar el modelo local.
        self.valloader = valloader

    # Este método es llamado cuando el servidor quiere obtener los parámetros actuales del modelo del cliente.
    # Utiliza la función get_parameters para extraer los pesos del modelo y enviarlos al servidor.
    # Los parámetros del modelo local se envían al servidor después del entrenamiento.
    def get_parameters(self, config):
        return get_parameters(self.net)


    def fit(self, parameters, config):
        # Actualiza el modelo del cliente con los parámetros enviados por el servidor.
        # Antes de entrenar, el cliente sincroniza su modelo con el modelo global del servidor.
        set_parameters(self.net, parameters)
        # Entrena el modelo local usando el conjunto de datos trainloader por una época.
        train(self.net, self.trainloader, epochs=1)
        # Obtiene los nuevos parámetros del modelo (actualizados después del entrenamiento)
        # para enviarlos de vuelta al servidor.
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        # El modelo local se sincroniza con los parámetros enviados por el servidor antes de la evaluación.
        set_parameters(self.net, parameters)
        # Evalúa el modelo local en el conjunto de validación (valloader) para obtener la pérdida y la precisión.
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

    #get_parameters: Envía los parámetros actuales del modelo al servidor.
    #fit: Entrena el modelo localmente con los datos del cliente y devuelve los parámetros actualizados.
    #evaluate: Evalúa el modelo con el conjunto de validación del cliente y devuelve la pérdida y la precisión.




# Esta función es utilizada para crear un cliente individual cuando se ejecuta el sistema federado.
# Cada cliente en el sistema tiene que ser instanciado correctamente, y esta función se encarga de eso.
# La función client_fn es una especie de fábrica de clientes.
def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Cargar el modelo:
    net = Net().to(DEVICE)

    # Se carga una partición específica de los datos de CIFAR-10.
    # Cada cliente trabaja con una partición distinta del conjunto de datos,
    # por lo que aquí se utiliza el partition_id para asignar una porción única de los datos a este cliente.
    partition_id = context.node_config["partition-id"]
    trainloader, valloader, _ = load_datasets(partition_id=partition_id)

    # Se instancia el cliente FlowerClient con su modelo, su conjunto de entrenamiento y su conjunto de validación
    return FlowerClient(net, trainloader, valloader).to_client()


# La clase FlowerClient: Define cómo un cliente individual en el sistema de aprendizaje federado
# entrena su modelo, evalúa su rendimiento y envía sus parámetros al servidor.

# La función client_fn: Se encarga de crear instancias de estos clientes con el modelo
# y los datos correspondientes a cada uno.

# La ClientApp: Es la aplicación del cliente en sí, que ejecuta el ciclo federado y se comunica con el servidor.


#  Esta ClientApp manejará el ciclo de entrenamiento federado desde el lado del cliente,
#  permitiendo que los clientes entrenen y evalúen sus modelos y se comuniquen con el servidor.
client = ClientApp(client_fn=client_fn)



# Define the Flower ServerApp
# we need to configure a strategy which encapsulates the federated learning approach/algorithm,
# for example, Federated Averaging (FedAvg)



# Esta función calcula el promedio ponderado de la métrica de precisión (accuracy) obtenida de los diferentes
# clientes que participan en la evaluación. Esto es importante porque cada cliente tiene un tamaño de conjunto
# de datos diferente y queremos que los clientes con más datos tengan más peso en el cálculo del promedio.
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}





# Create FedAvg strategy
# FedAvg (Federated Averaging) es una estrategia central en el aprendizaje federado,
# en la que el servidor central promedia los pesos de los modelos entrenados por los clientes.
# En lugar de enviar los datos al servidor, los clientes entrenan localmente y solo envían sus parámetros,
# que luego son agregados por el servidor para actualizar el modelo global.
strategy = FedAvg(
    fraction_fit=1.0,  # indica que el 100% de los clientes disponibles deben participar en cada ronda de entrenamiento
    fraction_evaluate=0.5,  # indica que el 50% de los clientes disponibles deben participar en cada ronda de evaluación
    min_fit_clients=10,  # el servidor nunca comenzará una ronda de entrenamiento con menos de 10 clientes disponibles
    min_evaluate_clients=5,  # asegura que el servidor no comenzará una ronda de evaluación con menos de 5 clientes conectados
    min_available_clients=10,  # El servidor requiere que haya al menos 10 clientes disponibles en total antes de iniciar el entrenamiento
    evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
    #cuando se evalúe la precisión (o cualquier otra métrica que puedas definir), el servidor utilizará
    # esta función para calcular un promedio ponderado de las métricas de precisión que reciben de los clientes.
)


# Este es el servidor central que coordina la comunicación y agregación entre los clientes.
# En esta función, configuras los componentes del servidor (como la estrategia y las rondas de entrenamiento)
# y los devuelves para que el servidor los utilice.
def server_fn(context: Context) -> ServerAppComponents:
   # 5 rondas de entrenamiento federado. Esto significa que el servidor realizará
   # cinco ciclos de comunicación con los clientes.

   # 1. Enviar el modelo global a los clientes.
   # 2. Los clientes entrenan localmente.
   # 3. Los clientes envían sus modelos actualizados al servidor.
   # 4. El servidor promedia los parámetros de los clientes y actualiza el modelo global.
    config = ServerConfig(num_rounds=5)

    return ServerAppComponents(strategy=strategy, config=config)


# Configuración de la estrategia: Definiste una estrategia FedAvg para seleccionar clientes y
# agregar sus modelos, especificando cuántos clientes deben participar en cada ronda de entrenamiento y evaluación.

# Configuración del servidor: Con server_fn, configuraste al servidor para realizar 5 rondas de aprendizaje federado.

# Creación del servidor: Finalmente, creaste la aplicación del servidor con ServerApp, que maneja todoa el ciclo de aprendizaje federado.


# Este código final crea la aplicación del servidor (ServerApp),
# que es el nodo central en el ciclo de aprendizaje federado
server = ServerApp(server_fn=server_fn)




# Run the training
# Este diccionario backend_config especifica los recursos que cada cliente utilizará durante la simulación.
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}


# Run simulation
# Este código está ejecutando una simulación, lo que significa que tanto el servidor como los clientes
# se ejecutan en el mismo entorno (como una máquina local)
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_CLIENTS,
    backend_config=backend_config,
)
############################# Step 2: Federated Learning with Flower ########################