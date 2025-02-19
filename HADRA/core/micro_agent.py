# HADRA/core/micro_agent.py

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class MicroAgent(Model):
    def __init__(self, state_dim, action_dim):
        """
        :param state_dim: Dimensión del estado (ej: [pos_x, pos_y, pos_z, goal_x, goal_y, goal_z]).
        :param action_dim: Dimensión de la acción (ej: velocidades en x, y, z).
        """
        super(MicroAgent, self).__init__()
        self.actor = self._build_actor(state_dim, action_dim)

    def _build_actor(self, state_dim, action_dim):
        inputs = layers.Input(shape=(state_dim,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.LayerNormalization()(x)
        # Salida normalizada entre -1 y 1, se puede escalar en función de límites reales de velocidad
        outputs = layers.Dense(action_dim, activation='tanh')(x)
        return Model(inputs, outputs)
    
    def call(self, inputs, **kwargs):
        # Cuando se llame al objeto Microagent, se utiliza la red actor interna
        return self.actor(inputs, **kwargs)

    def get_action(self, state, deterministic=True):
        """
        :param state: Estado actual del dron, ej: [pos_x, pos_y, pos_z, goal_x, goal_y, goal_z].
        :param deterministic: Si True, retorna la acción sin ruido (útil para despliegue).
        :return: Acción calculada (por ejemplo, velocidades en x, y, z).
        """
        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        action = self.actor(state_tensor)
        return action.numpy()[0]