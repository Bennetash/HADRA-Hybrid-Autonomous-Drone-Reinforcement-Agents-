# HADRA/scripts/mission_deploy.py

import argparse
import numpy as np
import tensorflow as tf
from HADRA.core.env import AirSimDroneEnv
from HADRA.core.hierarchical_agent import ColonyAgent
from HADRA.core.micro_agent import MicroAgent
import time

class MissionDeployer:
    def __init__(self, model_path=None):
        # Inicializar el entorno con 4 drones
        self.env = AirSimDroneEnv(["Drone1", "Drone2", "Drone3", "Drone4"])
        # Instanciar el Agente Colmena que genera sub-metas
        self.colony_agent = ColonyAgent(formation_shape="line", offset=1.0)
        # Definir dimensiones: estado = [pos_x, pos_y, pos_z, goal_x, goal_y, goal_z]
        state_dim = 6
        action_dim = 3  # vx, vy, vz
        self.micro_agents = {drone: MicroAgent(state_dim, action_dim) for drone in self.env.drones}

        # Cargar pesos preentrenados si se proporciona el directorio de modelos
        if model_path:
            for drone in self.env.drones:
                self.micro_agents[drone].load_weights(f"{model_path}/{drone}.h5")

    def run_mission(self, target_point):
        """
        Ejecuta la misión hasta que todos los drones alcancen el destino.
        
        :param target_point: Lista o array [x, y, z] que indica el destino global.
        """
        try:
            mission_complete = False
            while not mission_complete:
                # Obtener observaciones del entorno
                observations = self.env.get_observations()
                # Extraer posiciones actuales (se asume que los 3 primeros valores son [x, y, z])
                current_positions = {}
                for drone in observations:
                    current_positions[drone] = observations[drone][:3]

                # Generar sub-metas con el Agente Colmena basado en el destino global
                subgoals = self.colony_agent.generate_subgoals(current_positions, target_point)

                # Calcular acciones para cada dron: estado = [posición_actual, sub-meta]
                actions = {}
                for drone in self.env.drones:
                    state = np.concatenate([current_positions[drone], subgoals[drone]])
                    action = self.micro_agents[drone].get_action(state, deterministic=True)
                    actions[drone] = action

                # Ejecutar las acciones en el entorno
                self.env.execute_actions(actions)
                time.sleep(0.1)  # Ciclo de control

                # Verificar si todos los drones han alcanzado el destino (dentro de un umbral)
                mission_complete = self._check_mission_completion(current_positions, target_point)
        except KeyboardInterrupt:
            print("Interrupción manual: aterrizando drones...")
            self.env.reset()

    def _check_mission_completion(self, positions, target_point, threshold=1.0):
        """
        Verifica que todos los drones estén a menos de 'threshold' metros del destino.
        """
        for drone, pos in positions.items():
            if np.linalg.norm(np.array(pos) - np.array(target_point)) > threshold:
                return False
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, nargs=3, required=True, help="Coordenadas del destino [x, y, z]")
    parser.add_argument("--model-path", type=str, default=None, help="Ruta a los pesos de los modelos (opcional)")
    args = parser.parse_args()

    deployer = MissionDeployer(model_path=args.model_path)
    deployer.run_mission(target_point=args.target)