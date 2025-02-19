# HADRA/core/agent.py

import tensorflow as tf
from tensorflow.keras import layers, Model

class SACAgent(Model):
    def __init__(self, state_dim=10, action_dim=2):
        super().__init__()
        self.actor = self._build_actor(state_dim, action_dim)
        self.critic = self._build_critic(state_dim, action_dim)
        
    def _build_actor(self, state_dim, action_dim):
        inputs = layers.Input(shape=(state_dim,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.LayerNormalization()(x)
        mu = layers.Dense(action_dim, activation='tanh')(x)
        log_std = layers.Dense(action_dim)(x)
        return Model(inputs, [mu, log_std])
    
    def _build_critic(self, state_dim, action_dim):
        state_input = layers.Input(shape=(state_dim,))
        action_input = layers.Input(shape=(action_dim,))
        x = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(256, activation='relu')(x)
        q_value = layers.Dense(1)(x)
        return Model([state_input, action_input], q_value)
    
    def get_action(self, state, deterministic=False):
        mu, log_std = self.actor(state)
        if deterministic:
            return mu
        std = tf.exp(log_std) * 0.2
        return mu + tf.random.normal(mu.shape) * std