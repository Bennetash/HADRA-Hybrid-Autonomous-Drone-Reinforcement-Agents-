# HADRA/core/utils.py

import yaml
import json
import numpy as np
import csv
from datetime import datetime
from typing import Dict, Any, Callable

def load_config(path: str) -> Dict[str, Any]:
    """Carga un archivo YAML de configuración."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: Dict[str, Any], path: str):
    """Guarda un diccionario como archivo YAML."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def transform_coordinates(pos: np.ndarray, sim_scale: float = 100.0) -> np.ndarray:
    """Convierte coordenadas de AirSim (UE4) a metros reales."""
    return pos * sim_scale

def drone_logger(log_path: str = "logs/drone_metrics.csv") -> Callable:
    """Decorador para registrar métricas de los drones."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result = func(*args, **kwargs)
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, func.__name__, str(args), str(kwargs), str(result)])
            return result
        return wrapper
    return decorator

def quaternion_to_euler(x: float, y: float, z: float, w: float) -> tuple:
    """Convierte cuaterniones a ángulos de Euler (roll, pitch, yaw) en grados."""
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.degrees(np.arctan2(t0, t1))
    
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.degrees(np.arcsin(t2))
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(t3, t4))
    
    return (roll, pitch, yaw)

def check_airsim_connection(client, timeout=10):
    """Verifica la conexión con AirSim."""
    from time import time
    start = time()
    while time() - start < timeout:
        try:
            if client.ping():
                return True
        except:
            continue
    raise ConnectionError("No se pudo conectar a AirSim después de 10 segundos")