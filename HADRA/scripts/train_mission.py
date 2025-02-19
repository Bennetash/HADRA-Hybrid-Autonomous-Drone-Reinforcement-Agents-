import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from HADRA.core.env import AirSimDroneEnv
from HADRA.core.micro_agent import MicroAgent
from HADRA.core.hierarchical_agent import ColonyAgent
from HADRA.core.utils import load_config
import time
import os

# Parámetros RL globales
GAMMA = 0.99
TAU = 0.005

def build_critic(state_dim, action_dim):
    """Construye la red crítica que toma (estado, acción) y devuelve Q-value."""
    state_input = tf.keras.Input(shape=(state_dim,))
    action_input = tf.keras.Input(shape=(action_dim,))
    x = tf.keras.layers.Concatenate()([state_input, action_input])
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    q_value = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=[state_input, action_input], outputs=q_value)

def update_target_network(target, source, tau):
    new_weights = []
    for t, s in zip(target.get_weights(), source.get_weights()):
        new_weights.append(t * (1 - tau) + s * tau)
    target.set_weights(new_weights)

class MissionTrainer:
    def __init__(self, config_path="HADRA/configs/training.yaml"):
        self.config = load_config(config_path)
        # Inicializar entorno con 4 drones
        self.env = AirSimDroneEnv(["Drone1", "Drone2", "Drone3", "Drone4"])
        # Instanciar el Agente Colmena
        self.colony_agent = ColonyAgent(formation_shape="line", offset=1.0)
        # Dimensión de estado micro: [pos_x, pos_y, pos_z, goal_x, goal_y, goal_z] → 6
        # Dimensión de acción: [vx, vy, vz] → 3
        state_dim = 6
        action_dim = 3
        self.micro_agents = {drone: MicroAgent(state_dim, action_dim) for drone in self.env.drones}
        # Forza la construccion de los modelos de los microagentes
        dummy_input = tf.zeros((1, state_dim))
        for drone in self.env.drones:
            _ = self.micro_agents[drone](dummy_input)
        # Creamos la red crítica y su red objetivo para cada dron
        self.critics = {drone: build_critic(state_dim, action_dim) for drone in self.env.drones}
        self.target_critics = {drone: build_critic(state_dim, action_dim) for drone in self.env.drones}
        for drone in self.env.drones:
            self.target_critics[drone].set_weights(self.critics[drone].get_weights())
        # Optimizadores
        self.actor_optimizers = {drone: tf.keras.optimizers.Adam(3e-4) for drone in self.env.drones}
        self.critic_optimizers = {drone: tf.keras.optimizers.Adam(3e-4) for drone in self.env.drones}
        # Buffer de experiencias
        self.replay_buffers = {drone: [] for drone in self.env.drones}

    def preprocess_state(self, arr, expected_dim):
        """
        Transforma 'arr' en un vector unidimensional de tamaño 'expected_dim'.
        Si tiene menos elementos, rellena con ceros; si tiene más, recorta.
        """
        arr = np.array(arr).flatten()
        if arr.shape[0] < expected_dim:
            padding = np.zeros(expected_dim - arr.shape[0])
            arr = np.concatenate([arr, padding])
        elif arr.shape[0] > expected_dim:
            arr = arr[:expected_dim]
        return arr

    def _calculate_reward(self, current_pos, subgoal):
        """Recompensa negativa basada en la distancia entre la posición actual y la sub-meta."""
        return -np.linalg.norm(np.array(current_pos) - np.array(subgoal))
    
    def _store_experience(self, drone, state, action, reward, next_state):
        state = self.preprocess_state(state, 6)
        action = self.preprocess_state(action, 3)
        next_state = self.preprocess_state(next_state, 6)
        # Si la forma no es la esperada, se descarta
        if state.shape != (6,) or action.shape != (3,) or next_state.shape != (6,):
            print(f"[{drone}] Experiencia descartada: state {state.shape}, action {action.shape}, next_state {next_state.shape}")
            return
        self.replay_buffers[drone].append((state, action, reward, next_state))
        if len(self.replay_buffers[drone]) > self.config["training"]["buffer_size"]:
            self.replay_buffers[drone].pop(0)
    
    def _update_agents(self):
        batch_size = self.config["training"]["batch_size"]
        for drone in self.env.drones:
            if len(self.replay_buffers[drone]) < batch_size:
                continue
            indices = np.random.choice(len(self.replay_buffers[drone]), batch_size, replace=False)
            batch = [self.replay_buffers[drone][i] for i in indices]
            states, actions, rewards, next_states = zip(*batch)
            # Convertir a tensores
            states_tensor = tf.convert_to_tensor(np.stack(states, axis=0), dtype=tf.float32)
            actions_tensor = tf.convert_to_tensor(np.stack(actions, axis=0), dtype=tf.float32)
            rewards_tensor = tf.convert_to_tensor(np.array(rewards).reshape(-1, 1), dtype=tf.float32)
            next_states_tensor = tf.convert_to_tensor(np.stack(next_states, axis=0), dtype=tf.float32)
            
            # Actualizar el crítico
            with tf.GradientTape() as tape:
                next_actions = self.micro_agents[drone].actor(next_states_tensor)
                target_q = self.target_critics[drone]([next_states_tensor, next_actions])
                y = rewards_tensor + GAMMA * target_q
                q_values = self.critics[drone]([states_tensor, actions_tensor])
                critic_loss = tf.reduce_mean(tf.square(q_values - y))
            critic_grads = tape.gradient(critic_loss, self.critics[drone].trainable_variables)
            self.critic_optimizers[drone].apply_gradients(zip(critic_grads, self.critics[drone].trainable_variables))
            
            # Actualizar el actor
            with tf.GradientTape() as tape:
                actions_pred = self.micro_agents[drone].actor(states_tensor)
                actor_loss = -tf.reduce_mean(self.critics[drone]([states_tensor, actions_pred]))
            actor_grads = tape.gradient(actor_loss, self.micro_agents[drone].trainable_variables)
            self.actor_optimizers[drone].apply_gradients(zip(actor_grads, self.micro_agents[drone].trainable_variables))
            
            # Actualizar la red crítica objetivo
            update_target_network(self.target_critics[drone], self.critics[drone], TAU)
    
    def train(self, episodes=1000):
        # Nota: Se usa un destino con z negativo para que los drones no aterricen.
        target_point = [10.0, 10.0, -5.0]
        for episode in tqdm(range(episodes)):
            states = self.env.reset()
            episode_rewards = {drone: 0 for drone in self.env.drones}
            states_micro = {}
            for step in range(self.config["training"]["max_steps"]):
                current_positions = {}
                for drone in states:
                    current_positions[drone] = states[drone][:3]
                subgoals = self.colony_agent.generate_subgoals(current_positions, target_point)
                
                actions = {}
                for drone in self.env.drones:
                    s = np.concatenate([current_positions[drone], subgoals[drone]])
                    s = self.preprocess_state(s, 6)
                    states_micro[drone] = s
                    action = self.micro_agents[drone].get_action(s, deterministic=False)
                    actions[drone] = action
                next_states = self.env.execute_actions(actions)
                
                for drone in self.env.drones:
                    reward = self._calculate_reward(current_positions[drone], subgoals[drone])
                    next_state = np.concatenate([next_states[drone][:3], subgoals[drone]])
                    next_state = self.preprocess_state(next_state, 6)
                    self._store_experience(drone, states_micro[drone], actions[drone], reward, next_state)
                    episode_rewards[drone] += reward
                
                if step % 5 == 0:
                    self._update_agents()
                
                states = next_states
            
            if episode % 50 == 0:
                self._save_models(episode)
                print(f"Episode {episode}: Rewards {episode_rewards}")
    
    def _save_models(self, episode):
        save_dir = "data/pretrained"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for drone in self.env.drones:
            self.micro_agents[drone].save_weights(f"{save_dir}/{drone}_mission_ep{episode}.weights.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000, help="Número de episodios de entrenamiento")
    args = parser.parse_args()
    trainer = MissionTrainer()
    trainer.train(episodes=args.episodes)