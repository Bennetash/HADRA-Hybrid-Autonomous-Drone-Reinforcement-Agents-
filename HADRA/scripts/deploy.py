# HADRA/scripts/deploy.py

import argparse
import numpy as np
from HADRA.core.env import AirSimDroneEnv
from HADRA.core.agent import SACAgent

class HADRA_Deployer:
    def __init__(self, model_path):
        # Se utiliza un entorno con dos drones para este ejemplo.
        self.env = AirSimDroneEnv(["Drone1", "Drone2"])
        # Se instancian los agentes SAC para cada dron.
        self.agents = {
            "Drone1": SACAgent(256*144*3, 2),
            "Drone2": SACAgent(256*144*3, 2)
        }
        self._load_models(model_path)

    def _load_models(self, path):
        for drone in self.env.drones:
            self.agents[drone].load_weights(f"{path}/{drone}.h5")

    def run_formation(self):
        try:
            while True:
                states = self.env.get_observations()
                actions = {}
                for drone in self.env.drones:
                    state = states[drone].flatten()
                    action = self.agents[drone].get_action(state[np.newaxis], deterministic=True)
                    actions[drone] = action.numpy()[0]
                self.env.execute_actions(actions)
        except KeyboardInterrupt:
            print("Aterrizando drones...")
            self.env.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Ruta a los modelos preentrenados")
    args = parser.parse_args()

    deployer = HADRA_Deployer(args.model_path)
    deployer.run_formation()