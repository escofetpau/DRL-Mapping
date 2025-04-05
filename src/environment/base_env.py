from typing import Optional
import gymnasium as gym
import sparse
from gymnasium import spaces
import numpy as np
import wandb
from abc import abstractmethod


from src.utils.constants import N_CORES, CORE_CAPACITY, ACTION_TYPES
from random import seed
from data.circuit_generator import generate_circuit
from src.environment.utils import (
    is_last_qbit,
    unpack_action,
    is_qbit_placed,
)


seed(42)


class BaseGraphSeriesEnv(gym.Env):
    def __init__(
        self,
        circuit_config,
        action_type,
        weights_reward,
        mask_full_cores=True,
        n_cores=N_CORES,
        core_capacity=CORE_CAPACITY,
    ):
        """
        config: Ara mateix, és només el circuit. La resta de paràmetres s'agafen de src.utils.constants
        n_qbits: Es dedueix directament del circuit
        observation_space: Definim l'espai d'observació com a:
            node_features: Matriu de dim [n_qbits, node_features] que serà paddejada a [max_qbits, node_features]
                Serà redimensionada al feature extractor, però la dimensió s'ha de definir amb el padding ja fet.
            adj_matrix: Matriu d'adjacència [max_qbits, max_qbits] on un valor != 0 indica aresta. El seu valor és el lookahead (o derivats)
                De nou, serà redimensionada al feature_extractor. Si es vol afegir una altra edge_feature, probablement n'hi ha prou amb
                fer-la de dimensió [2, max_qbits, max_qbits], on la primera dimensió indica la feature que s'està mirant.
            n_qbits: Indica el nombre real de qbits a utilitzar, el valor que utilitza el feature extractor per a redimensionar la observació.
        action_space: En el meu cas, el nombre de cores, al teu, també serà spaces.Discrete però posant self.n_qbits*N_CORES (si ho tinc ben entès)
        """
        super().__init__()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert action_type in ACTION_TYPES, f"action_type should be in {ACTION_TYPES}"

        self.weights_reward = weights_reward
        self.action_type = action_type
        self.mask_full_cores = mask_full_cores

        self.random_circuits = circuit_config.get("random_circuits")
        if self.random_circuits:
            circuit = generate_circuit(
                circuit_config["n_slices"],
                circuit_config["n_qbits"],
                circuit_config["gates_per_slice"],
            )
            self.gates_per_slice = circuit_config["gates_per_slice"]
        else:
            circuit = sparse.COO(np.array(circuit_config["circuit"]))

        self.n_qbits = circuit_config["n_qbits"]
        self.n_cores = n_cores
        self.n_slices = circuit.shape[0]  # Important for the end of episode

        self.observation_space = spaces.Dict(
            {
                "old_allocation": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.n_qbits, self.n_cores),
                    dtype=np.int32,
                ),
                "new_allocation": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.n_qbits, self.n_cores),
                    dtype=np.int32,
                ),
                "interactions": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.n_qbits, self.n_qbits),
                    dtype=np.int32,
                ),
                "lookaheads": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(self.n_qbits, self.n_qbits),
                    dtype=np.float32,
                ),
                "core_capacities": spaces.Box(
                    low=0, high=np.inf, shape=(self.n_cores,), dtype=np.float32
                ),
                "n_qbits": spaces.Box(
                    low=1, high=self.n_qbits, shape=(1,), dtype=np.int32
                ),
                "flag": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.n_qbits, 1),
                    dtype=np.int32,
                ),
            }
        )

        self.circuit = circuit # TODO: check that no self-loops
        self.all_lookaheads = self._get_lookaheads(circuit)

        self.old_allocation = np.zeros((self.n_qbits, self.n_cores))
        self.new_allocation = np.zeros((self.n_qbits, self.n_cores))
        self.interactions = circuit[0].todense()
        self.lookahead = self.all_lookaheads[0]
        self.core_capacities = np.array([core_capacity] * n_cores)

        self.action_space = (
            spaces.Discrete(self.n_cores)
            if self.action_type == 'S'
            else spaces.Discrete(self.n_qbits * self.n_cores)
        )

        self.qbit_idx = 0 if self.action_type == 'S' else None # only needed if action_type == 'S'

        self.nl_com = 0
        self.intervention = 0
        self.direct_capacity_violation = 0

        self.reset()

    @abstractmethod
    def _take_action(self, action: int) -> tuple[int, float, bool, bool, bool]:
        pass

    def _get_lookaheads(self, circuit, sigma=1):
        """Get lookahead tensor from the circuit

        Returns:
            lookahead: np.array of shape (n_slices, n_qbits, n_qbits)
        """
        lookahead = np.zeros((self.n_slices, self.n_qbits, self.n_qbits))
        lookahead[-1] = circuit[-1].todense()

        for i in range(len(circuit) - 2, -1, -1):
            lookahead[i] = 2 ** (-sigma) * lookahead[i + 1] + circuit[i].todense()

        return lookahead

    def _get_observation(self):
        """
        Com que s'ha definit observation_space com un spaces.Dict, es retorna la
        observació com un diccionari amb els paràmetres indicats a la definició
        """
        flag_vector = np.zeros((self.n_qbits,1))
        flag_vector[self.qbit_idx] = 1

        obs = {
            "old_allocation": self.old_allocation,
            "new_allocation": self.new_allocation,
            "interactions": self.interactions,
            "lookaheads": self.lookahead,
            "n_qbits": np.array([self.n_qbits]),
            "core_capacities": self.core_capacities,
            "flag": flag_vector
        }
        return obs

    def _get_reward(self, nl_com: int, intervention: bool, direct_capacity_violation: bool):
        self.nl_com = nl_com
        self.intervention = intervention
        self.direct_capacity_violation = direct_capacity_violation
        reward = -self.weights_reward["nonlocal"] * nl_com - self.weights_reward[
            "intervention"
        ] * int(intervention)
        return reward

    def _set_new_placement(self, qbit, core):
        self.new_allocation[qbit][core] = 1

    def step(self, action: int):
        """Action: {}"""
        actual_action, nl_com, intervention, direct_capacity_violation, truncated = (
            self._take_action(action=action)
        )

        if truncated:
            return self._get_observation(), -100, False, True, {}

        qbit, core = unpack_action(actual_action, self.qbit_idx, self.action_type, self.n_cores)
        reward = self._get_reward(nl_com, intervention, direct_capacity_violation)

        self.total_reward += reward
        self.total_nl_coms += nl_com
        self.total_interventions += int(intervention)
        self.total_direct_capacity_violations += int(direct_capacity_violation)

        done = False

        #Can I just use the qbit_idx? Do not think this is replicable for both action types
        if is_last_qbit(qbit, self.new_allocation):
            done = self.slice_idx == self.n_slices - 1  # if last slice then end episode
            if not done:
                self._advance_to_next_slice()

        if self.action_type == 'S': # si faig una acció sergi, si o si haure alocat aquell qbit
            while self.qbit_idx < self.n_qbits - 1 and is_qbit_placed(self.qbit_idx, self.new_allocation):
                self.qbit_idx += 1

        info = (
            {}
        )  # info s'utilitza per a donar feedback a l'usuari. PPO el demana encara que estigui buit.

        return self._get_observation(), reward, done, truncated, info


    def _set_slice(self):
        """Sets the current slice based on self.slice_idx"""

        # Nodes
        self.old_allocation = (
            np.zeros((self.n_qbits, self.n_cores))
            if self.slice_idx == 0
            else self.new_allocation
        )

        if self.action_type == 'S':
            self.qbit_idx = 0

        self.new_allocation = np.zeros((self.n_qbits, self.n_cores))  # (q, c)

        # Edges
        self.interactions = self.circuit[self.slice_idx].todense()
        self.lookahead = self.all_lookaheads[self.slice_idx]

        self.core_capacities = np.array([CORE_CAPACITY] * N_CORES)

    def _advance_to_next_slice(self):
        self.slice_idx += 1
        self._set_slice()

    def reset(self, seed=None, options=None):

        if self.random_circuits:
            circuit = generate_circuit(
                self.n_slices, self.n_qbits, self.gates_per_slice
            )
            self.circuit = circuit
            self.all_lookaheads = self._get_lookaheads(circuit)

        self.total_reward = 0
        self.total_nl_coms = 0
        self.total_interventions = 0
        self.total_direct_capacity_violations = 0

        self.slice_idx = 0
        self._set_slice()

        info = (
            {}
        )  # info s'utilitza per a donar feedback a l'usuari. PPO el demana encara que estigui buit.

        # Reset ha de retornar la observació
        return self._get_observation(), info

    
    @abstractmethod
    def env_mask(self):
        pass
