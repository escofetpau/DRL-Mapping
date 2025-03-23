import pytest
from unittest.mock import patch

import numpy as np
from sparse import COO

from src.environment.base_env import BaseGraphSeriesEnv

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


@pytest.fixture
def circuit_config():
    return {
        "n_slices": 3,
        "n_qbits": 3,
        "gates_per_slice": [1],
        "circuit": None,
        "random_circuits": True
    }

@pytest.fixture
def base_env(circuit_config):
    # Creamos la instancia sin llamar al __init__ original
    env = BaseGraphSeriesEnv.__new__(BaseGraphSeriesEnv)
   
    # Configuramos manualmente los atributos que necesitamos para la prueba
    env.action_type = 'S'
    env.n_slices = circuit_config["n_slices"]
    env.n_qbits = circuit_config["n_qbits"]

    return env


def test_get_lookaheads(base_env):
    base_env.n_slices = 3
    base_env.n_qbits = 3
    
    circuit = [
        COO.from_numpy(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])),  # Matriz para el primer slice
        COO.from_numpy(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])),  # Matriz para el segundo slice
        COO.from_numpy(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]))   # Matriz para el tercer slice
    ]

    target_lookahead = np.array([
        [[0.5, 1, 0], [0.25, 0, 0], [0, 0, 0]],  # Matriz para el primer slice
        [[1, 0, 0], [0.5, 0, 0], [0, 0, 0]],     # Matriz para el segundo slice
        [[0, 0, 0], [1, 0, 0], [0, 0, 0]]        # Matriz para el tercer slice
    ])
    
    lookahead = base_env._get_lookaheads(circuit, sigma=1)

    assert lookahead.shape == (base_env.n_slices, base_env.n_qbits, base_env.n_qbits)
    
    assert np.allclose(target_lookahead, lookahead, rtol=1e-5, atol=1e-8)

