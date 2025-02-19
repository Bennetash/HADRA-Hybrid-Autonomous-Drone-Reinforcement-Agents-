# HADRA/scripts/monitor.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import airsim
import numpy as np
from threading import Thread
from HADRA.core.utils import quaternion_to_euler   
import time
import csv
import os
from datetime import datetime

LOG_FILE = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'logs', 'telemetry_log.csv.csv')

# Si el archivo no existe, crea el archivo y escribe la cabecera
if not os.path.exists(os.path.dirname(LOG_FILE)):
    os.makedirs(os.path.dirname(LOG_FILE))

if not os.path.isfile(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Drone", "Posicion", "Velocidad", "Orientacion"])

def log_telemetry(drone, position, velocity, orientation):
    """Escribe una linea en el archivo CSV con la telemetria del dron."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, drone, position, velocity, orientation])

class AirSimMonitor:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.drones = ["Drone1", "Drone2", "Drone3", "Drone4"]
        self.running = True
        self.data = {
            "positions": {},
            "velocities": {},
            "orientations": {}
        }

    def _telemetry_thread(self):
        while self.running:
            for drone in self.drones:
                state = self.client.getMultirotorState(vehicle_name=drone)
                kinematics = state.kinematics_estimated
                
                position = (
                    kinematics.position.x_val,
                    kinematics.position.y_val,
                    kinematics.position.z_val
                )
                velocity = (
                    kinematics.linear_velocity.x_val,
                    kinematics.linear_velocity.y_val,
                    kinematics.linear_velocity.z_val
                )
                q = kinematics.orientation
                orientation = quaternion_to_euler(q.x_val, q.y_val, q.z_val, q.w_val)

                self.data["positions"][drone] = position
                self.data["velocities"][drone] = velocity
                self.data["orientations"][drone] = orientation

                log_telemetry(drone, position, velocity, orientation)
            time.sleep(0.1)
                

    def display_telemetry(self):
        while self.running:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n=== TELEMETRÍA EN TIEMPO REAL ===")
            for drone in self.drones:
                print(f"\n[{drone}]")
                print(f"Posición: {self.data['positions'].get(drone, 'N/A')}")
                print(f"Velocidad: {self.data['velocities'].get(drone, 'N/A')}")
                print(f"Orientación: {self.data['orientations'].get(drone, 'N/A')}")
            time.sleep(1)

    def start(self):
        Thread(target=self._telemetry_thread, daemon=True).start()
        self.display_telemetry()

if __name__ == "__main__":
    monitor = AirSimMonitor()
    try:
        monitor.start()
    except KeyboardInterrupt:
        monitor.running = False