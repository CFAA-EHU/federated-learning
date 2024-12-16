import flwr as fl
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from collections import OrderedDict
from typing import List, Tuple


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define la estrategia que utilizará el servidor, por ejemplo, FedAvg
strategy = FedAvg(
    fraction_fit=1.0,  # El 100% de los clientes participarán en cada ronda de >
    fraction_evaluate=0.5,  # Solo el 50% de los clientes participarán en la ev>
    min_fit_clients=2,  # Se necesitan al menos 2 clientes para entrenar
    min_evaluate_clients=1,  # Al menos 1 cliente para evaluar
    min_available_clients=2,  # Se requieren al menos 2 clientes disponibles en>
    evaluate_metrics_aggregation_fn=weighted_average,
)

# Iniciar el servidor de Flower
if __name__ == "__main__":
    # Inicia el servidor con la estrategia definida
    fl.server.start_server(
        server_address="10.98.101.104:8080",  # o poner 0.0.0.0:8080
        config=fl.server.ServerConfig(num_rounds=5),  # Configuración del servidor con 5 rondas
        strategy=strategy,  # Estrategia FedAvg
    )
