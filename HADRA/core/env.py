# HADRA/core/env.py

import airsim
import numpy as np
from typing import Dict, List

class AirSimDroneEnv:
    def __init__(self, drone_names: List[str] = ["Drone1", "Drone2", "Drone3", "Drone4"]):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.drones = drone_names
        self._setup_drones()
        
    def _setup_drones(self):
        for drone in self.drones:
            self.client.enableApiControl(True, drone)
            self.client.armDisarm(True, drone)
            self.client.takeoffAsync(vehicle_name=drone).join()
    
    def get_observations(self) -> Dict[str, np.ndarray]:
        obs = {}
        for drone in self.drones:
            state = self.client.getMultirotorState(vehicle_name=drone)
            kinematics = state.kinematics_estimated

            obs[drone] = np.array([
                kinematics.position.x_val, kinematics.position.y_val, kinematics.position.z_val,
                kinematics.linear_velocity.x_val, kinematics.linear_velocity.y_val, kinematics.linear_velocity.z_val,
                kinematics.orientation.x_val, kinematics.orientation.y_val,
                kinematics.orientation.z_val, kinematics.orientation.w_val
            ])
        return obs
    
    def execute_actions(self, actions: Dict[str, np.ndarray]):
        # Ejecuta las acciones enviadas a cada dron.
        # Se asume que action es un vector con [vx, vy, vz].
        for drone, action in actions.items():
            vx, vy, vz = float(action[0]), float(action[1]), float(action[2])
            self.client.moveByVelocityAsync(vx, vy, vz, 1.0, vehicle_name=drone)
        # Retorna las observaciones después de ejecutar la acción.
        return self.get_observations()
    
    def reset(self):
        for drone in self.drones:
            self.client.reset()
        # Reconfigurar y despegar de nuevo a los drones
        self._setup_drones()
        return self.get_observations()