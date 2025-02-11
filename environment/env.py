from typing import Optional
import gymnasium as gym
#import torch
from gymnasium import spaces
import numpy as np
from src.utils.constants import *
from random import sample, seed
from data.circuit_generator import generate_circuit
import wandb


ACTION_TYPES = ['sequential', 'qubits x cores'] # [sergi, laia]

seed(42)

class GraphSeriesEnv(gym.Env):
    def __init__(self, config, action_type = 'qubits x cores', mask_full_cores = True, n_cores = N_CORES, weights_reward = {'nonlocal': 1, 'capacity': 30, 'intervention': 40, 'slice_idx': 50}):
        '''
        Funció init del environment.
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
        '''
        super(GraphSeriesEnv, self).__init__()
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert action_type in ACTION_TYPES, f'action_space should be in {ACTION_TYPES}'

        self.weights_reward = weights_reward
        self.action_type = action_type
        self.mask_full_cores = mask_full_cores

        self.random_circuits = config.get('random_circuits')
        if self.random_circuits:
            circuit = generate_circuit(config['n_slices'], config['n_qubits'], config['gates_per_slice'])
            self.gates_per_slice = config['gates_per_slice']
        else:
            circuit = config['circuit']
        
        self.n_qbits = circuit[0].shape[0]
        self.n_cores = n_cores

        # TODO: passar qbit_reservations com a observació
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_qbits, 2*N_CORES), dtype=np.float32), #TODO: canviar a int
            "adj_matrix": spaces.Box(low = 0, high = np.inf, shape=(2, self.n_qbits, self.n_qbits), dtype = np.float32),
            "core_capacities": spaces.Box(low = 0, high = np.inf, shape = (N_CORES,), dtype=np.float32),
            "n_qbits": spaces.Box(low=1, high=self.n_qbits, shape=(1,), dtype=np.float32) # TODO: canviar a int
        })
        
        
        self.action_space = spaces.Discrete(self.n_cores) if self.action_type == ACTION_TYPES[0] else spaces.Discrete(self.n_qbits*self.n_cores)


        self.n_slices = circuit.shape[0] # Important for the end of episode
        self.circuit = circuit
        self.lookahead = self._get_lookahead(circuit)

        self.reset() # Initialize les node_features and adjacency matrix


    def _get_lookahead(self, circuit, func='exp', sigma=1):
        '''Get lookahead tensor from the circuit

        Returns:
            lookahead: np.array of shape (n_slices, n_qbits, n_qbits)
        '''
        assert func == 'exp', 'For the time being only exp is implemented'
        
        lookahead = np.zeros((self.n_slices, self.n_qbits, self.n_qbits))
        lookahead[-1] = circuit[-1].todense()

        for i in range(len(circuit) - 2, -1, -1):
            lookahead[i] = 2**(-sigma) * lookahead[i+1] + circuit[i].todense()    

        return lookahead 

    
    def _is_last_qbit(self, qbit):
        new_allocation = self.node_features[:self.n_qbits, self.n_cores : self.n_cores*2]
        return all(any(new_allocation[q]) for q in range(self.n_qbits) if q != qbit)


    def _get_observation(self):
        '''
        Com que s'ha definit observation_space com un spaces.Dict, es retorna la observació com un diccionari amb els paràmetres
        indicats a la definició. Posat en una funció per nitidesa.

        '''
        obs = {'node_features': self.node_features, 'adj_matrix': self.adj_matrix, 'n_qbits': np.array([self.n_qbits]), 'core_capacities': self.core_capacities}
        return obs


    def _calculate_reward(self, action, nl_com, intervention, direct_capacity_violation):
        weights = self.weights_reward
        self.nl_com = nl_com
        self.intervention = intervention
        self.direct_capacity_violation = direct_capacity_violation
        reward = - weights['nonlocal']*nl_com - weights['intervention']*int(intervention) \
                - weights['capacity']*int(direct_capacity_violation) #+ weights['slice_idx']*self.slice_idx / self.n_slices  
        
        return reward 

    def _unpack_action(self, action):
        '''action = qbit * self.n_cores + core'''
        qbit = action // self.n_cores if self.action_type == ACTION_TYPES[1] else self.qbit_idx
        core = action % self.n_cores if self.action_type == ACTION_TYPES[1] else action
        return qbit, core
    
    def _has_pair(self, qbit):
        return sum(self.adj_matrix[0][qbit]) > 0

    def _is_qbit_placed_vectorized(self):
        new_allocation = self.node_features[:self.n_qbits, self.n_cores : self.n_cores * 2]
        return np.any(new_allocation, axis=1)
    
    def _set_new_placement(self, action):
        qbit, core = self._unpack_action(action)
        self.node_features[qbit][self.n_cores + int(core)] = 1

    def _take_action(self, action: int) -> tuple[int, bool]:
        #L'acció ubica el core a la new_core_allocation de la acció. És a step que marquem el següent qbit a allocar.
        # action = self.check_action_valid(action)
                
        def is_direct_capacity_violation(core):
            return self.core_capacities[core] == 0

        def is_interaction_violation(qbit, core):
            return self.qbit_reservations[qbit] == -1 or self.qbit_reservations[qbit] == core

        def is_missing_space_for_interaction_violation(qbit, core):
            return self._has_pair(qbit) and self.core_capacities[core] < 2

        def is_direct_reservation_capacity_violation(qbit, core):
            return self.core_reservations[core] > 0 and self.core_capacities[core] - self.core_reservations[core] < 1

        def is_interaction_reservation_capacity_violation(qbit, core):
            return self.core_reservations[core] > 0 and self.core_capacities[core] - self.core_reservations[core] - self._has_pair(qbit) < 1

        def is_no_space_for_future_gates_violation(qbit, core):
            # TODO: REVISAR LES MASKS
            aux = self.core_capacities.copy()
            aux[core] -= 1 # suposem que coloquem qbit a core
            full_gate_capacity = np.sum((aux-self.core_reservations)//2)

            if self.action_type == ACTION_TYPES[0]: # sergi
                mask = (np.arange(self.adj_matrix.shape[1]) > self.qbit_idx) & (self.qbit_reservations == -1)
                remaining_full_gates = np.sum(self.adj_matrix[0,mask,:]) // 2

            elif self.action_type == ACTION_TYPES[1]: # laia
                future_qbits_mask = (self.qbit_reservations == -1) & (~self._is_qbit_placed_vectorized()) & (np.arange(self.n_qbits) != qbit) 
                remaining_full_gates = np.sum(self.adj_matrix[0][:, future_qbits_mask]) // 2

            
            return full_gate_capacity < remaining_full_gates

        def get_violations(qbit, core):
            # DUBTE: CAPACITY VIOLATION PROVOCARA MOLTES ALTRES VIOLATIONS. COM HO COMPTABILITZO?
            direct_capacity_violation = is_direct_capacity_violation(core)
            interaction_violation = is_interaction_violation(qbit, core)
            missing_space_for_interaction_violation = is_missing_space_for_interaction_violation(qbit, core)
            direct_reservation_capacity_violation = is_direct_reservation_capacity_violation(qbit, core)
            interaction_reservation_capacity_violation = is_interaction_reservation_capacity_violation(qbit, core)
            no_space_for_future_gates = is_no_space_for_future_gates_violation(qbit, core)
            
            return direct_capacity_violation, interaction_violation, missing_space_for_interaction_violation, direct_reservation_capacity_violation, interaction_reservation_capacity_violation, no_space_for_future_gates
        
        
        def get_valid_action(qbit) -> int | None:
            ''''''
            if self.qbit_reservations[qbit] != -1: # si té reserva, l'acció és la reserva
                return qbit*self.n_cores + self.qbit_reservations[qbit] if self.action_type == ACTION_TYPES[1] else self.qbit_reservations[qbit]
            return next((qbit*self.n_cores + core if self.action_type == ACTION_TYPES[1] else core for core in sample(range(self.n_cores), self.n_cores) if not any(get_violations(qbit, core))), None)


        def update_reservations(action: int):
            '''En una mateixa slice, un qubit no tindrà més de 1 interacció'''
            qbit, core = self._unpack_action(action)
            intended_allocation = self.qbit_reservations[qbit]
            if intended_allocation != -1: # if there was a reservation of current qbit, remove it
                self.core_reservations[intended_allocation] -= 1
            elif self._has_pair(qbit):
                neighbour = np.argmax(self.adj_matrix[0][qbit])
                if not self._is_qbit_placed(neighbour):
                    self.core_reservations[core] += 1
                    self.qbit_reservations[neighbour] = core
                    
        intervention = False
        qbit, core = self._unpack_action(action)

        violations = get_violations(qbit, core)

        if not any(violations):
            intervention = True
            #print('invalid action', action)
            actual_action: Optional[int] = get_valid_action(qbit)

            if actual_action is None: # TODO: remove because truncation never happens
                print('SHA TRUNCAT, FATAL!!')
                return None, True, True, True, True
        else:
            actual_action = action

        qbit, core = self._unpack_action(actual_action)
        self.core_capacities[core] -= 1
        update_reservations(actual_action)
        self._set_new_placement(actual_action)

        old_placement = self.node_features[qbit][:self.n_cores]
        new_placement = self.node_features[qbit][self.n_cores : 2*self.n_cores]

        nl_comm = sum(abs(old_placement - new_placement)) // 2

        return actual_action, nl_comm, intervention, violations[0], False


    def step(self, action: int):
        '''Action: {}'''
        actual_action, nl_com, intervention, direct_capacity_violation, truncated = self._take_action(action)
        
        
        if truncated:
            return self._get_observation(), -100, False, True, {}
        
        qbit, core = self._unpack_action(actual_action)
        reward = self._calculate_reward(actual_action, nl_com, intervention, direct_capacity_violation)

        self.total_reward += reward
        self.total_nl_coms += nl_com
        self.total_interventions += int(intervention)
        self.total_direct_capacity_violations += int(direct_capacity_violation)
        
        done = False
       
        if self._is_last_qbit(qbit):
            done = self.slice_idx == self.n_slices - 1 # if last slice then end episode
            if not done:
                self._advance_to_next_slice() #Si l'episodi no ha acabat, preparem la observació per a representar la següent slic


        info = {} #info s'utilitza per a donar feedback a l'usuari. PPO el demana encara que estigui buit.

        #Ara mateix no utilitzem truncated però s'ha de retornar igual. Es podria afegir, per exemple, si he sobrepassat la capacitat d'un core
        #de manera que PPO deixi d'executar l'episodi actual. (Possible) TODO
        
        
        return self._get_observation(), reward, done, truncated, info

    def _set_slice(self):
        '''Sets the current slice based on self.slice_idx'''

        # Nodes
        if self.slice_idx == 0: #La core allocation és nul·la al principi del circuit
            core_allocation = np.zeros((self.n_qbits, self.n_cores))
        else:
            core_allocation = self.node_features[:self.n_qbits, self.n_cores : self.n_cores*2] # la que tenia o 0 depenent de si he fet reset
        
        new_core_allocation = np.zeros((self.n_qbits, self.n_cores)) # one-hot

        self.node_features = np.concatenate([core_allocation, new_core_allocation], axis=1)
        
        
        # Edges
        circuit_slice = self.circuit[self.slice_idx].todense()
        lookahead = self.lookahead[self.slice_idx]
        
        circuit_expanded = np.expand_dims(circuit_slice, axis=0)  # shape (q, q, 1)
        lookahead_expanded = np.expand_dims(lookahead, axis=0)  # shape (q, q, 1)

        self.adj_matrix = np.concatenate([circuit_expanded, lookahead_expanded], axis=0)  # shape (2*q, q, 1)
        #print(self.adj_matrix)
        self.core_capacities = np.array([CORE_CAPACITY] * N_CORES)

        #Relocation purposes
        self.qbit_reservations = np.array([-1]*self.n_qbits)
        self.core_reservations = np.array([0] * N_CORES)


    def _advance_to_next_slice(self):
        self.slice_idx += 1
        self._set_slice()


    def reset(self, seed = None, options = None):
        
        if self.random_circuits:
            circuit = generate_circuit(self.n_slices, self.n_qbits, self.gates_per_slice)
            self.circuit = circuit
            self.lookahead = self._get_lookahead(circuit)

        self.total_reward = 0
        self.total_nl_coms = 0
        self.total_interventions = 0
        self.total_direct_capacity_violations = 0

        self.slice_idx = 0  #Porta el compte de quina slice del circuit s'està mirant.
        self._set_slice()

        info = {} #info s'utilitza per a donar feedback a l'usuari. PPO el demana encara que estigui buit.

        #Reset ha de retornar la observació com a tal. Després de realitzar els canvis de reinici. Busco observació.
        return self._get_observation(), info


    def _is_qbit_placed(self, qbit):
        #self.node_features = np.concatenate([core_allocation, new_core_allocation], axis=1)
        new_allocation = self.node_features[:self.n_qbits, self.n_cores : self.n_cores*2]

        return any(new_allocation[qbit])

    def qbit_mask(self): 
        mask = np.ones(self.n_qbits*self.n_cores, dtype=bool) if self.action_type == ACTION_TYPES[1] else np.ones(self.n_cores, dtype=bool)
        
        if self.action_type == ACTION_TYPES[1]:
            for qbit in range(self.n_qbits): # ban qbits already placed
                if self._is_qbit_placed(qbit):
                    for core in range(self.n_cores):
                        mask[qbit*self.n_cores + core] = False

        if self.mask_full_cores:
            for core, capacity in enumerate(self.core_capacities): # ban full cores
                if capacity == 0:
                    for qbit in range(self.n_qbits):
                        if self.action_type == ACTION_TYPES[1]:
                            mask[qbit*self.n_cores + core] = False
                        else:
                            mask[core] = False

        return mask
    
    
