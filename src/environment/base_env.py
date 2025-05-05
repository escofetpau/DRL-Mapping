import gymnasium as gym
import sparse
from gymnasium import spaces
import numpy as np
from abc import abstractmethod


from random import seed
from src.circuit_generator.random_circuit_generator import generate_circuit
from src.environment.utils import (
    is_last_qubit,
    unpack_action,
    is_qubit_placed,
)


seed(42)


class BaseGraphSeriesEnv(gym.Env):
    def __init__(
        self,
        circuit_config: dict,
        action_type: str,
        weights_reward: dict,
        n_qubits: int,
        n_cores: int,
        core_capacity: int | None = None,

    ):
        """
        Args:
            - circuit_config: Configuració del circuit. Pot ser un circuit random o un circuit predefinit.
            - action_type: Can be 'S' (sequentially allocate each qubit to a core aka Sergi) or 'L' (choose qubit and core aka Laia)
            - weights_reward
            - n_qubits
            - n_cores
            - core_capacity: If None, uses n_qubits / n_cores. If not None, uses core_capacity.
        """
        super().__init__()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert action_type in ['S', 'L'], f"action_type should be in ['S', 'L'] but got {action_type}"

        if core_capacity is None:
            core_capacity = n_qubits // n_cores

        self.action_type = action_type
        self.weights_reward = weights_reward
        self.n_qubits = n_qubits
        self.n_cores = n_cores
        self.core_capacity = core_capacity

        self.random_circuits = circuit_config.get("random_circuits")
        if self.random_circuits:
            circuit = generate_circuit(
                n_qubits,
                circuit_config["n_slices"],
                circuit_config["gates_per_slice"],
            )
            self.gates_per_slice = circuit_config["gates_per_slice"]
        else:
            circuit = sparse.COO(np.array(circuit_config["circuit"]))

        self.n_slices = circuit.shape[0]  # Important for the end of episode

        self.observation_space = spaces.Dict(
            {
                "old_allocation": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.n_qubits, self.n_cores),
                    dtype=np.int32,
                ),
                "new_allocation": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.n_qubits, self.n_cores),
                    dtype=np.int32,
                ),
                "interactions": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.n_qubits, self.n_qubits),
                    dtype=np.int32,
                ),
                "lookaheads": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(self.n_qubits, self.n_qubits),
                    dtype=np.float32,
                ),
                "core_capacities": spaces.Box(
                    low=0, high=np.inf, shape=(self.n_cores,), dtype=np.float32
                ),
                "n_qubits": spaces.Box(
                    low=1, high=self.n_qubits, shape=(1,), dtype=np.int32
                ),
                "flag": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.n_qubits, 1),
                    dtype=np.int32,
                ),
            }
        )

        self.circuit = circuit  # TODO: check that no self-loops
        assert circuit.shape == (self.n_slices, self.n_qubits, self.n_qubits), (
            f"Expected circuit shape {(self.n_slices, self.n_qubits, self.n_qubits)} but got {circuit.shape}"
        )

        self.all_lookaheads = self._get_all_lookaheads(circuit) # s'inicialitzen al reset
        self.lookahead = self.all_lookaheads[0]

        self.old_allocation = np.zeros((self.n_qubits, self.n_cores))
        self.new_allocation = np.zeros((self.n_qubits, self.n_cores))
        self.interactions = circuit[0].todense()
        self.core_capacities = np.array([core_capacity] * n_cores)

        self.action_space = (
            spaces.Discrete(self.n_cores)
            if self.action_type == 'S'
            else spaces.Discrete(self.n_qubits * self.n_cores)
        )

        self.qubit_idx = 0 # only needed if action_type == 'S'

        self.nl_com = 0
        self.intervention = 0
        self.direct_capacity_violation = 0
        self.missing_space_for_interaction_violation = 0
        self.no_space_for_future_gates_violation = 0

        self.slice_idx = 0
        self._set_slice(slice_idx=0)


    @abstractmethod
    def _take_action(self, action: int) -> tuple[int, bool]:
        pass

    def _get_all_lookaheads(self, circuit: sparse.COO, sigma=1) -> np.ndarray:
        """Get lookahead tensor from the circuit

        Returns:
            lookahead: np.array of shape (n_slices, n_qubits, n_qubits)
        """
        lookahead = np.zeros((self.n_slices, self.n_qubits, self.n_qubits))
        lookahead[-1] = circuit[-1].todense()

        for i in range(len(circuit) - 2, -1, -1):
            lookahead[i] = 2 ** (-sigma) * lookahead[i + 1] + circuit[i].todense()

        return lookahead

    def _get_observation(self) -> dict:
        """
        Com que s'ha definit observation_space com un spaces.Dict, es retorna la
        observació com un diccionari amb els paràmetres indicats a la definició
        """
        flag_vector = np.zeros((self.n_qubits, 1))
        flag_vector[self.qubit_idx] = 1

        obs = {
            "old_allocation": self.old_allocation,
            "new_allocation": self.new_allocation,
            "interactions": self.interactions,
            "lookaheads": self.lookahead,
            "n_qubits": np.array([self.n_qubits]),
            "core_capacities": self.core_capacities,
            "flag": flag_vector,
        }
        return obs

    def _get_reward(self) -> int:
        weights = self.weights_reward
        intervention = self.intervention
        direct_capacity_violation = self.direct_capacity_violation
        missing_space_for_interaction_violation = (
            self.missing_space_for_interaction_violation
        )
        no_space_for_future_gates_violation = self.no_space_for_future_gates_violation
        nl_com = self.nl_com

        reward = (
            - weights["nonlocal"] * nl_com
            - weights["intervention"] * int(intervention)
            - weights["direct_capacity_violation"] * direct_capacity_violation
            - weights["missing_space_for_interaction_violation"]
            * missing_space_for_interaction_violation
            - weights["no_space_for_future_gates_violation"]
            * no_space_for_future_gates_violation
        )
        return reward

    def _set_new_placement(self, qubit: int, core: int) -> None:
        self.new_allocation[qubit][core] = 1

    def step(self, action: int) -> tuple[dict, int, bool, bool, dict]:
        """Action: {}"""
        actual_action, truncated = self._take_action(action=action)

        if truncated:
            return self._get_observation(), -100, False, True, {}

        qubit, core = unpack_action(
            actual_action, self.qubit_idx, self.action_type, self.n_cores
        )
        reward = self._get_reward()

        done = False

        # Can I just use the qubit_idx? Do not think this is replicable for both action types
        if is_last_qubit(qubit, self.new_allocation):
            done = self.slice_idx == self.n_slices - 1  # if last slice then end episode
            if not done:
                self._set_slice(self.slice_idx + 1)

        if self.action_type == "S":
            while self.qubit_idx < self.n_qubits - 1 and is_qubit_placed(
                self.qubit_idx, self.new_allocation
            ):
                self.qubit_idx += 1

        info = (
            {}
        )  # info s'utilitza per a donar feedback a l'usuari. PPO el demana encara que estigui buit.

        return self._get_observation(), reward, done, truncated, info

    def _set_slice(self, slice_idx: int) -> None:
        """Sets the current slice based on self.slice_idx"""

        self.slice_idx = slice_idx
        self.qubit_idx = 0 # only needed if action_type == 'S'

        # Nodes
        self.old_allocation = (
            np.zeros((self.n_qubits, self.n_cores))
            if slice_idx == 0
            else self.new_allocation
        )

        self.new_allocation = np.zeros((self.n_qubits, self.n_cores))  # (q, c)

        # Edges
        self.interactions = self.circuit[slice_idx].todense()
        self.lookahead = self.all_lookaheads[slice_idx]

        self.core_capacities = np.array([self.core_capacity] * self.n_cores)


    def reset(self, seed=None, options=None) -> tuple[dict, dict]:

        if self.random_circuits:
            circuit = generate_circuit(
                self.n_qubits, self.n_slices, self.gates_per_slice
            )
            self.circuit = circuit
            self.all_lookaheads = self._get_all_lookaheads(circuit)

        self.nl_com = 0
        self.intervention = 0
        self.direct_capacity_violation = 0
        self.missing_space_for_interaction_violation = 0
        self.no_space_for_future_gates_violation = 0

        self._set_slice(0)

        info = (
            {}
        )  # info s'utilitza per a donar feedback a l'usuari. PPO el demana encara que estigui buit.

        return self._get_observation(), info

    @abstractmethod
    def env_mask(self) -> None:
        pass
