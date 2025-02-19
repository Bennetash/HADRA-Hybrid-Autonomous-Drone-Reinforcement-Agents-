# HADRA/scripts/train_mission.py

import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from HADRA.core.env import AirSimDroneEnv
from HADRA.core.micro_agent import MicroAgent
from HADRA.core.hierarchical_agent import ColonyAgent
from HADRA.core.utils import load_config
import time

class MissionTrainer:
    def __init__(self, config_path="HADRA/configs/training.yaml"):
        self.config = load_config(config_path)
        # Inicializar entorno con 4 drones
        self.env = AirSimDroneEnv(["Drone1", "Drone2", "Drone3", "Drone4"])
        # Instanciar el Agente Colmena
        self.colony_agent = ColonyAgent(formation_shape="line", offset=1.0)
        # Estado = [pos_x, pos_y, pos_z, goal_x, goal_y, goal_z] -> (6,)
        # Accion = [vel_x, vel_y, vel_z] -> (3,)
        state_dim = 6
        action_dim = 3
        self.micro_agents = {drone: MicroAgent(state_dim, action_dim) for drone in self.env.drones}
        self.optimizers = {drone: tf.keras.optimizers.Adam(3e-4) for drone in self.env.drones}
        # Buffer de experiencias individual para cada dron
        self.replay_buffers = {drone: [] for drone in self.env.drones}

    def preprocess_state(self, arr, expected_dim):
        """
        Transforma 'arr' en un vector unidimensional de tamaño 'expected_dim'
        Si 'arr' tiene menos elementos, se le hace padding con ceros
        Si 'arr' tiene mas elementos, se recorta
        """
        arr = np.array(arr).flatten()
        if arr.shape[0] < expected_dim:
            padding = np.zeros(expected_dim - arr.shape[0])
            arr = np.concatenate([arr, padding])
        elif arr.shape[0] > expected_dim:
            arr = arr[:expected_dim]
        return arr
    
    def _calculate_reward(self, current_pos, subgoal):
        """Recompensa negativa proporcional a la distancia entre la posición y la sub-meta."""
        return -np.linalg.norm(np.array(current_pos) - np.array(subgoal))
    
    def _store_experience(self, drone, state, action, reward, next_state):
        # Se procesa los arrays para asegurar la forma homogenea.
        state = self.preprocess_state(state, 6)
        action = self.preprocess_state(action, 3)
        next_state = self.preprocess_state(next_state, 6)

        # Depuracion: Imprime las formas y algunos valores representativos
        print(f"[{drone}] state shape: {state.shape}, action shape: {action.shape}, next_state shape: {next_state.shape}")
        print(f"State sample: {state[:5]}  |  Next state sample: {next_state[:5]}")

        if state.shape != (6,) or action.shape != (3,) or next_state.shape != (6,):
            print(f"[{drone}] Experiencia descartada: state {state.shape}, action {action.shape}, next_state {next_state.shape}")
            return

        # Añadir la experiencia al buffer de experiencias
        self.replay_buffers[drone].append((state, action, reward, next_state))
        # Mantener el buffer de experiencias dentro de un tamaño máximo
        if len(self.replay_buffers[drone]) > self.config["training"]["buffer_size"]:
            self.replay_buffers[drone].pop(0)
    
    def _update_agents(self):
        batch_size = self.config["training"]["batch_size"]
        for drone in self.env.drones:
            if len(self.replay_buffers[drone]) < batch_size:
                continue
            # Seleccionar indices aleatorios
            indices = np.random.choice(len(self.replay_buffers[drone]), batch_size, replace=False)
            batch = [self.replay_buffers[drone][i] for i in indices]
            states, actions, rewards, next_states = zip(*batch)
            # se usa np.stack para forzar la forma correcta
            try:
                states = np.stack(states, axis=0)
                actions = np.stack(actions, axis=0)
            except Exception as e:
                print(f"Error al apilar datos para {drone}: {e}")
                continue
            rewards = np.array(rewards).reshape(-1, 1)
            next_states = np.stack(next_states, axis=0)
            
            with tf.GradientTape() as tape:
                # Ejemplo simple: minimizar la diferencia entre la acción predicha y la acción ejecutada.
                predicted_actions = self.micro_agents[drone].actor(states)
                loss = tf.reduce_mean(tf.square(predicted_actions - actions))
            grads = tape.gradient(loss, self.micro_agents[drone].trainable_variables)
            self.optimizers[drone].apply_gradients(zip(grads, self.micro_agents[drone].trainable_variables))
    
    def train(self, episodes=1000):
        # Definir un destino global fijo para la misión durante el entrenamiento.
        target_point = [10.0, 10.0, 0.0]
        for episode in tqdm(range(episodes)):
            states = self.env.reset()
            episode_rewards = {drone: 0 for drone in self.env.drones}

            # Diccionario para almacenar el estado (para el microagente) de cada dron
            states_micro = {}
            
            for step in range(self.config["training"]["max_steps"]):
                # Extraer posiciones actuales (los primeros 3 valores) de cada dron
                current_positions = {}
                for drone in states:
                    current_positions[drone] = states[drone][:3]
                
                # Generar sub-metas con el Agente Colmena basado en el destino global
                subgoals = self.colony_agent.generate_subgoals(current_positions, target_point)
                
                actions = {}
                for drone in self.env.drones:
                    # Estado del microagente: concatenacion posicion actual y sub-meta debe tener 6 elementos
                    s = np.concatenate([current_positions[drone], subgoals[drone]])
                    s = self.preprocess_state(s, 6) # Aseguramos que el estado tenga 6 elementos
                    states_micro[drone] = s
                    action = self.micro_agents[drone].get_action(s, deterministic=False)
                    actions[drone] = action
                
                next_states = self.env.execute_actions(actions)
                
                # Calcular la recompensa para cada dron y almacenar la experiencia
                for drone in self.env.drones:
                    reward = self._calculate_reward(current_positions[drone], subgoals[drone])
                    # Estado siguiente: concatenacion posicion siguiente y sub-meta debe tener 6 elementos
                    next_state = np.concatenate([next_states[drone][:3], subgoals[drone]])
                    next_state = self.preprocess_state(next_state, 6) # Aseguramos que el estado tenga 6 elementos
                    self._store_experience(drone, states_micro[drone], actions[drone], reward, next_state)
                    episode_rewards[drone] += reward
                
                if step % 5 == 0:
                    self._update_agents()
                
                states = next_states
            
            if episode % 50 == 0:
                self._save_models(episode)
                print(f"Episode {episode}: Rewards {episode_rewards}")
    
    def _save_models(self, episode):
        for drone in self.env.drones:
            self.micro_agents[drone].save_weights(f"HADRA/data/pretrained/{drone}_mission_ep{episode}.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000, help="Número de episodios de entrenamiento")
    args = parser.parse_args()

    trainer = MissionTrainer()
    trainer.train(episodes=args.episodes)