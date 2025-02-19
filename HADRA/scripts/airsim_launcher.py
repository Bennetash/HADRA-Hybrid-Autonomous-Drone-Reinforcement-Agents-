# HADRA/scripts/airsim_launcher.py

import airsim
import json
import time

def initialize_simulation():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    # Cargar configuración desde airsim.json
    with open("C:/Users/GH/Documents/GitHub/HADRA_GPT/HADRA/configs/airsim.json", "r") as f:
        config = json.load(f)
    
    # Inicializar y armar cada dron según la configuración
    for vehicle_name in config["Vehicles"]:
        client.enableApiControl(True, vehicle_name)
        client.armDisarm(True, vehicle_name)
        client.takeoffAsync(vehicle_name=vehicle_name).join()
    
    print("Simulación iniciada en el mapa Blocks!")

if __name__ == "__main__":
    initialize_simulation()