import flwr as fl
from flwr.server.strategy import FedAvg

# Define la estrategia que utilizará el servidor
strategy = FedAvg(
    fraction_fit=1.0,  # El 100% de los clientes participarán en cada ronda de entrenamiento
    fraction_evaluate=0.5,  # Solo el 50% de los clientes participarán en la evaluación
    min_fit_clients=2,  # Se necesitan al menos 2 clientes para entrenar
    min_evaluate_clients=1,  # Al menos 1 cliente para evaluar
    min_available_clients=2,  # Se requieren al menos 2 clientes disponibles en total
)

# Iniciar el servidor de Flower
if __name__ == "__main__":
    print("Iniciando el servidor de Flower...")
    fl.server.start_server(
        server_address="10.98.101.104:8080",  # Dirección del servidor
        config=fl.server.ServerConfig(num_rounds=5),  # Configuración del servidor con 5 rondas
        strategy=strategy,  # Estrategia FedAvg
    )
    print("Servidor de Flower iniciado correctamente.")
