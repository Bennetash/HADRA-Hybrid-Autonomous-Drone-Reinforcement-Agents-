# HADRA/scripts/train.py

import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from HADRA.core.env import AirSimDroneEnv
from HADRA.core.agent import SACAgent
from HADRA.core.utils import load_config

class HADRA_Trainer:
    def __init__(self, config_path="HADRA/configs/training.yaml"):
        self.config = load_config(config_path)
        # Se utiliza un entorno con 4 drones
        self.env = AirSimDroneEnv(["Drone1", "Drone2", "Drone3", "Drone4"])
        self.agents = {
            "Drone1": SACAgent(256*144*3, 2),
            "Drone2": SACAgent(256*144*3, 2),
            "Drone3": SACAgent(256*144*3, 2),
            "Drone4": SACAgent(256*144*3, 2)
        }
        self.optimizers = {
            "Drone1": tf.keras.optimizers.Adam(3e-4),
            "Drone2": tf.keras.optimizers.Adam(3e-4),
            "Drone3": tf.keras.optimizers.Adam(3e-4),
            "Drone4": tf.keras.optimizers.Adam(3e-4)
        }
        self.replay_buffer = []

    def _store_experience(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))
        if len(self.replay_buffer) > self.config["training"]["buffer_size"]:
            self.replay_buffer.pop(0)

    def _update_agents(self):
        batch_size = self.config["training"]["batch_size"]
        if len(self.replay_buffer) < batch_size:
            return
        batch = np.random.choice(self.replay_buffer, batch_size, replace=False)
        states, actions, rewards, next_states = zip(*batch)
        
        for drone in ["Drone1", "Drone2", "Drone3", "Drone4"]:
            with tf.GradientTape() as tape:
                # Ejemplo simple: minimizar la diferencia entre acción predicha y acción ejecutada.
                predicted_actions = self.agents[drone].actor(np.array(states))
                loss = tf.reduce_mean(tf.square(predicted_actions - np.array(actions)))
            grads = tape.gradient(loss, self.agents[drone].trainable_variables)
            self.optimizers[drone].apply_gradients(zip(grads, self.agents[drone].trainable_variables))

    def train(self, episodes=1000):
        for episode in tqdm(range(episodes)):
            states = self.env.reset()
            total_rewards = {drone: 0 for drone in self.env.drones}

            for _ in range(self.config["training"]["max_steps"]):
                actions = {}
                for drone in self.env.drones:
                    state = states[drone].flatten()
                    action = self.agents[drone].get_action(state[np.newaxis])
                    actions[drone] = action
                next_states = self.env.execute_actions(actions)
                
                # Aquí se define una función de recompensa a modo de ejemplo.
                rewards = {}
                for drone in self.env.drones:
                    # Recompensa negativa proporcional a la norma de la acción (como ejemplo)
                    rewards[drone] = -np.linalg.norm(actions[drone])
                
                for drone in self.env.drones:
                    self._store_experience(
                        states[drone].flatten(),
                        actions[drone],
                        rewards[drone],
                        next_states[drone].flatten()
                    )
                    total_rewards[drone] += rewards[drone]

                self._update_agents()
                states = next_states

            if episode % 50 == 0:
                self._save_models(episode)
                print(f"Episode {episode}: Rewards {total_rewards}")

    def _save_models(self, episode):
        for drone in self.env.drones:
            self.agents[drone].save_weights(f"HADRA/data/pretrained/{drone}_ep{episode}.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000, help="Número de episodios de entrenamiento")
    args = parser.parse_args()

    trainer = HADRA_Trainer()
    trainer.train(episodes=args.episodes)