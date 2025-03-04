import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Data, Batch
from torch_geometric.nn.models import GCN

from src.models.gnn.gatv2 import GATv2Model
from src.utils.constants import N_CORES, ST_OUTPUT_DIM, ST_HIDDEN_DIM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNNFeatureExtractor(BaseFeaturesExtractor):
    '''
    El feature extractor s'encarrega de redimensionar les observacions i fer el forward per la GNN.
    '''
    def __init__(self, observation_space, gnn_name, edge_dim=2,
                 hidden_features=64, out_features=16,
                 num_heads=1, hidden_layers=4, dropout=0):
        assert gnn_name in ['GATv2', 'GCN'], f'Invalid GNN model name: {gnn_name}'

        super(GNNFeatureExtractor, self).__init__(observation_space, 
                                                   features_dim=out_features * 8 + N_CORES)

        in_features = observation_space['new_allocation'].shape[1] # PREGUNTA PER SERGI: n_qbits?? TODO: fix
        self.out_features = out_features

        self.gnn_name = gnn_name

        if self.gnn_name == 'GATv2':
            self.gnn = GATv2Model(in_features, edge_dim, hidden_features, out_features, num_heads, hidden_layers, dropout)
        else:
            self.gnn = GCN(in_features, hidden_features, hidden_layers, out_features, dropout)
        self.gnn.to(device)


    def _get_node_features(self, obs) -> np.ndarray:
        '''
        Returns the node features for the GNN, which are the new and old allocations.

        Arguments:
            - obs['new_allocation']: (batch_size, n_qbits, n_cores)
            - obs['old_allocation']: (batch_size, n_qbits, n_cores)

        Returns:
            - node_features: (batch_size, 2*n_qbits, n_qbits)

        '''
        n_qbits = int(obs['n_qbits'][0][0])
        return np.concat([obs['new_allocation'][:, :n_qbits, :n_qbits], obs['old_allocation'][:, :n_qbits, :n_qbits]], axis=1)


    def _get_adj_matrix(self, obs) -> np.ndarray:
        '''
        Returns the adjacency matrix for the GNN, 
        
        Arguments (no pooling): 
            - obs['interactions']: (batch_size, n_qbits, n_qbits)
            - obs['lookahead']: (batch_size, n_qbits, n_qbits)

        Returns:
            - adj_matrix: (batch_size, 2*n_qbits, n_qbits)
        '''
        n_qbits = int(obs['n_qbits'][0][0])
        return np.concat([obs['interactions'][:, :n_qbits, :n_qbits], obs['lookaheads'][:, :n_qbits, :n_qbits]], axis=1)


    def forward(self, obs):
        #Durant l'exploració és 1, durant l'entrenament es BATCH_SIZE
        batch_size = obs['new_allocation'].shape[0]
        #El retall s'haurà de fer dins del batcher una vegada hi hagi diversos nombres de qbits dins d'un batch
        #Ara agafo n_qbits del primer circuit però això no serà sempre veritat. TODO
        n_qbits = int(obs['n_qbits'][0][0])
        x = self._get_node_features(obs)
        adj_mat = self._get_adj_matrix(obs)

        #Passo pel batcher per a construir un sol estat no connex que contingui tots els estats
        batch = self.batcher(batch_size, x, adj_mat)

        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_features = batch.edge_attr.to(device)

        if self.gnn_name == 'GCN':
            edge_features = edge_features.sum(axis=1) # combine interaction and lookahead for GCN

        node_embeddings = self.gnn(x, edge_index, edge_features)
        #Quan hi hagi diversos nombres de qbits, s'haurà de refer utilitzant el batch.batch, que indica a quina observació
        #Pertany cada node TODO

        node_embeddings = node_embeddings.reshape(batch_size, n_qbits*self.out_features) #Batched flatten
        node_embeddings = torch.cat((node_embeddings, obs['core_capacities']), dim = 1)
        return node_embeddings
        '''
        Ara s'estan aplanant les node_embeddings, amb el set transformer el procés canviarà
        '''
        node_embeddings = node_embeddings.unsqueeze(0)

        #El que abans era un graf tot junt ara es torna a separar per instàncies dins del batch
        node_embeddings = node_embeddings.view(batch_size, n_qbits, self.out_features)

        return node_embeddings


    def batcher(self, batch_size, batch_node_features, batch_adj_matrix):
        '''
        Constructs a PyG.Data object for each graph observation and combines them into a batch.
        Handles multiple edge attributes (e.g., multiple features per edge).
        '''
        data_list = []

        # TODO: vectorize?

        for i in range(batch_size):
            # Node features for the i-th graph
            node_features = batch_node_features[i]  # Shape: [n_nodes, node_feature_dim]

            # Adjacency matrix for the i-th graph, with multiple edge attributes
            adj_matrix = batch_adj_matrix[i]  # Shape: [num_edge_features, n_nodes, n_nodes]

            # Compute edge_index from the union of nonzero entries across all features
            combined_adj_matrix = adj_matrix.sum(axis=0)  # Shape: [n_nodes, n_nodes]
            edge_index = torch.nonzero(combined_adj_matrix, as_tuple=False).t()  # Shape: [2, n_edges]

            # Extract edge attributes for all features
            edge_attr = adj_matrix[:, edge_index[0], edge_index[1]].t()  # Shape: [n_edges, num_edge_features]

            # Create a PyG Data object for the current graph
            data = Data(
                x=node_features,  # Node features
                edge_index=edge_index,  # Edge indices
                edge_attr=edge_attr  # Edge features (one column per edge feature)
            )
            data_list.append(data)

        # Combine all graphs into a single batch
        batch = Batch.from_data_list(data_list)
        return batch
