#!/usr/bin/env python3
"""load the FrozenLakeEnv environment from gymnasium"""


import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the FrozenLakeEnv environment from gymnasium based on the provided parameters.
    
    Parameters:
    - desc (list of lists or None): Custom map description.
    - map_name (str or None): Pre-made map name (e.g., '4x4' or '8x8').
    - is_slippery (bool): Whether the ice is slippery.
    
    Returns:
    - env: The loaded Gymnasium FrozenLake environment.
    """
    if desc is None and map_name is None:
        desc = generate_random_map(size=8)
        
    env = gym.make('FrozenLake-v1', desc=desc, map_name=map_name, is_slippery=is_slippery)
    return env