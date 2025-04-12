import numpy as np
import pytest
from src.environment.utils import is_missing_space_for_interaction_violation, is_no_space_for_future_gates_violation


@pytest.mark.parametrize("qubit, core, core_caps, interactions, expected", [
    (0, 0, [1, 2], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], False), # no té parella
    (2, 0, [1, 2], [[0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], True), # cap el qubit, pero no la parella
    (2, 1, [1, 2], [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], False), # caben el qubit i la parella
])
def test_is_missing_space_for_interaction_violation(qubit, core, core_caps, interactions, expected):
    assert is_missing_space_for_interaction_violation(qubit, core, core_caps, np.array(interactions)) == expected


@pytest.mark.parametrize("qubit, core, core_capacities, new_allocation, interactions, expected", [
    (1, 1, [1, 2], [[1, 0], [0, 0], [0, 0], [0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], True),
    (1, 0, [1, 2], [[1, 0], [0, 0], [0, 0], [0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], False),
    (0, 1, [1, 2], [[0, 0], [0, 0], [0, 0], [1, 0]], [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], True),
    (0, 0, [2, 1], [[0, 0], [0, 0], [0, 0], [0, 1]], [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], True),
])
def test_is_no_space_for_future_gates_violation(qubit, core, 
    core_capacities,
    new_allocation,
    interactions,
    expected):
    assert is_no_space_for_future_gates_violation(qubit, core,  core_capacities, np.array(new_allocation), np.array(interactions)) == expected