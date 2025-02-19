# HADRA/core/hierarchical_agent.py

import numpy as np

class ColonyAgent:
    def __init__(self, formation_shape="line", offset=1.0):
        """
        :param formation_shape: Tipo de formación ('line', 'square', etc.).
        :param offset: Distancia de separación entre drones.
        """
        self.formation_shape = formation_shape
        self.offset = offset

    def generate_subgoals(self, current_positions, target_point):
        """
        Genera sub-metas para cada dron dado su estado actual y el objetivo global.
        
        :param current_positions: dict { drone_name: np.array([x, y, z]) }
        :param target_point: list o np.array [x, y, z] con el destino global.
        :return: dict { drone_name: np.array([x, y, z]) } con la sub-meta para cada dron.
        """
        num_drones = len(current_positions)
        # Calcular el centro de la formación actual
        center = np.mean(np.array(list(current_positions.values())), axis=0)
        # Vector de dirección global desde el centro hasta el objetivo
        direction = np.array(target_point) - center
        norm = np.linalg.norm(direction)
        if norm == 0:
            direction_norm = np.zeros_like(direction)
        else:
            direction_norm = direction / norm
        
        # Para formación en línea: se distribuyen los drones lateralmente respecto a la dirección de avance
        lateral_direction = np.array([-direction_norm[1], direction_norm[0], 0])
        sorted_drones = sorted(current_positions.keys())
        subgoals = {}
        for idx, drone in enumerate(sorted_drones):
            # Calcular el offset lateral
            offset_vector = (idx - (num_drones - 1) / 2) * self.offset * lateral_direction
            # La sub-meta es el objetivo global ajustado con el offset lateral
            subgoals[drone] = np.array(target_point) + offset_vector
        return subgoals