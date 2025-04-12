import numpy as np

def get_nl_com(old_allocation, new_allocation, qubit):
    return np.sum(np.abs(old_allocation[qubit] - new_allocation[qubit])) // 2


def has_pair(qubit, interactions):
    return sum(interactions[qubit]) > 0


def is_qubit_placed(qubit, new_allocation):
    return any(new_allocation[qubit])


def is_qubit_placed_vectorized(new_allocation):
    return np.any(new_allocation, axis=1)


def is_last_qubit(qubit, new_allocation):
    n_qubits = new_allocation.shape[0]
    return all(any(new_allocation[q]) for q in range(n_qubits) if q != qubit)


def unpack_action(action, curr_qubit_idx, action_type, n_cores):
    """action = qubit * n_cores + core"""
    qubit = action // n_cores if action_type == 'L' else curr_qubit_idx
    core = action % n_cores if action_type == 'L' else action
    return qubit, core


def pack_action(qubit, core, action_type, n_cores):
    if action_type == 'L':
        return qubit * n_cores + core
    return core


### Violations
def is_direct_capacity_violation(core, core_capacities):
    return core_capacities[core] == 0


def is_interaction_violation(qubit, core, qubit_reservations):
    return qubit_reservations[qubit] != -1 and qubit_reservations[qubit] != core


def is_missing_space_for_interaction_violation(qubit, core, core_capacities, interactions):
    return has_pair(qubit, interactions) and core_capacities[core] < 2


def is_direct_reservation_capacity_violation(core, core_capacities, core_reservations):
    return (
        core_reservations[core] > 0
        and core_capacities[core] - core_reservations[core] < 1
    )


def is_interaction_reservation_capacity_violation(
    qubit, core, interactions, core_capacities, core_reservations
):
    return (
        core_reservations[core] > 0
        and core_capacities[core] - core_reservations[core] - has_pair(qubit, interactions) < 1
    )


def is_no_space_for_future_gates_violation(
    qubit,
    core,
    core_capacities,
    new_allocation,
    interactions,
    core_reservations=None,
    qubit_reservations=None,
):
    n_qubits = new_allocation.shape[0]
    aux = core_capacities.copy()
    aux[core] -= 1  # suposem que coloquem qubit a core
    full_gate_capacity = (
        np.sum((aux - core_reservations) // 2)
        if core_reservations is not None
        else np.sum(a=np.array(aux) // 2)
    )

    future_qubits_mask = (
        (~is_qubit_placed_vectorized(new_allocation))
        & (np.arange(n_qubits) != qubit)
    )
    if qubit_reservations is not None:
        future_qubits_mask = future_qubits_mask & (qubit_reservations == -1)

    remaining_full_gates = np.sum(interactions[:n_qubits][future_qubits_mask]) // 2

    return full_gate_capacity < remaining_full_gates
