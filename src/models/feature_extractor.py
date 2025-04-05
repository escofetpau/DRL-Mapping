import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Data, Batch
from torch_geometric.nn.models import GCN, GAT

from src.models.gnn.gatv2 import GATv2Model
from src.utils.constants import N_CORES, ST_OUTPUT_DIM, ST_HIDDEN_DIM


class GNNFeatureExtractor(BaseFeaturesExtractor):
    '''
    El feature extractor s'encarrega de redimensionar les observacions i fer el forward per la GNN.
    '''
    def __init__(self, observation_space, action_type, gnn_name, device,
                 hidden_features, n_qbits, out_features=16,
                 num_heads=1, hidden_layers=4, dropout=0):
        assert gnn_name in ['GATv2', 'GCN'], f'Invalid GNN model name: {gnn_name}'

        super().__init__(observation_space,
                                                   features_dim=out_features * n_qbits + N_CORES)
        self.device = device
        self.action_type = action_type
        self.out_features = out_features
        self.gnn_name = gnn_name

        in_features = 2*N_CORES + 1 if self.action_type == 'S' else 2*N_CORES
        
        if self.gnn_name == 'GATv2':
            self.gnn = GAT(in_features, hidden_features, hidden_layers, out_features, dropout, v2 = True, edge_dim = 2)

        else:
            self.gnn = GCN(in_features, hidden_features, hidden_layers, out_features, dropout, edge_dim = 1)
    
        self.gnn.to(self.device)


    def _get_node_features(self, obs) -> torch.Tensor:
        '''
        Returns the node features for the GNN, which are the new and old allocations.

        Arguments:
            - obs['new_allocation']: torch.Tensor (batch_size, n_qbits, n_cores)
            - obs['old_allocation']: torch.Tensor (batch_size, n_qbits, n_cores)
            - obs['flag']: (batch_size, n_qbits, 1) if action = 'S'

        Returns:
            - node_features: (batch_size, n_qbits, 2*n_cores + 1) / (batch_size, n_qbits, 2*n_cores)

        '''
        print('new_allocation type', type(obs['new_allocation']))
        n_qbits = int(obs['n_qbits'][0][0])
        features = torch.cat([obs['new_allocation'][:, :n_qbits, :n_qbits], obs['old_allocation'][:, :n_qbits, :n_qbits]], dim=2)

        if self.action_type == 'S':
            features = torch.cat([features, obs['flag']], dim=2)

        if self.device == 'cuda':
            features = features.cuda()

        return features

    def forward(self, obs):
        batch_size = obs['new_allocation'].shape[0]
        n_qbits = int(obs['n_qbits'][0][0])
        x = self._get_node_features(obs)

        #Passo pel batcher per a construir un sol estat no connex que contingui tots els estats
        batch = self.batcher(x, obs['interactions'], obs['lookaheads'])

        x = batch.x.to(self.device)
        edge_index = batch.edge_index.to(self.device) # (2, n_edges)
        edge_features = batch.edge_attr.to(self.device) # (n_edges, n_edges_features)
 
        if self.gnn_name == 'GCN':
            edge_features = edge_features.sum(axis=1)

        node_embeddings = self.gnn(x, edge_index, edge_features)
        #Quan hi hagi diversos nombres de qbits, s'haurà de refer utilitzant el batch.batch, que indica a quina observació
        #Pertany cada node TODO

        node_embeddings = node_embeddings.reshape(batch_size, n_qbits*self.out_features) #Batched flatten
        node_embeddings = torch.cat((node_embeddings, obs['core_capacities']), dim = 1)
        print('output feature extractor', node_embeddings.shape)
        return node_embeddings
        '''
        Ara s'estan aplanant les node_embeddings, amb el set transformer el procés canviarà
        '''
        node_embeddings = node_embeddings.unsqueeze(0)

        #El que abans era un graf tot junt ara es torna a separar per instàncies dins del batch
        node_embeddings = node_embeddings.view(batch_size, n_qbits, self.out_features)

        return node_embeddings

    def batcher(self, batch_node_features, batch_interactions, batch_lookaheads):
        '''
        Constructs a PyG.Batch object from a batch of graphs with two edge types (interaction & lookahead).
        '''
        B, N, _ = batch_interactions.shape
        x = batch_node_features.view(-1, batch_node_features.size(-1))  # [B*N, F]
        node_batch = torch.arange(B).repeat_interleave(N)               # [B*N]

        if self.gnn_name == 'GATv2':
            edges1 = (batch_interactions != 0).nonzero(as_tuple=False)  # [E1, 3]
            edges2 = (batch_lookaheads != 0).nonzero(as_tuple=False)    # [E2, 3]

            edge_attr1 = batch_interactions[edges1[:, 0], edges1[:, 1], edges1[:, 2]].unsqueeze(1)
            edge_attr2 = batch_lookaheads[edges2[:, 0], edges2[:, 1], edges2[:, 2]].unsqueeze(1)

            edge_index1 = edges1[:, 1:] + edges1[:, 0].unsqueeze(1) * N
            edge_index2 = edges2[:, 1:] + edges2[:, 0].unsqueeze(1) * N

            edge_index = torch.cat([edge_index1, edge_index2], dim=0).t()  # [2, E]
            edge_attr = torch.cat([edge_attr1, edge_attr2], dim=0)         # [E, 1]

        elif self.gnn_name == 'GCN':
            combined_adj = batch_interactions + batch_lookaheads
            edges = (combined_adj != 0).nonzero(as_tuple=False)  # [E, 3]
            edge_attr = combined_adj[edges[:, 0], edges[:, 1], edges[:, 2]].unsqueeze(1)
            edge_index = edges[:, 1:] + edges[:, 0].unsqueeze(1) * N
            edge_index = edge_index.t()  # [2, E]

        # Construir el batch directamente
        batch = Batch(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=node_batch
        )

        return batch


    def batcher2(self, batch_size, batch_node_features, batch_interactions, batch_lookaheads):
        '''
        Constructs a PyG.Data object for each graph observation and combines them into a batch.
        Handles multiple edge attributes (e.g., multiple features per edge).
        '''
        data_list = []

        for i in range(batch_size):
            node_features = batch_node_features[i]  # Shape: [n_qbits, 2*n_cores (+1)]
            interaction = batch_interactions[i]
            lookahead = batch_lookaheads[i]

            if self.gnn_name == 'GATv2':
                edge_index1 = (interaction != 0).nonzero(as_tuple=False).t()  # Shape: [2, num_edges1]
                edge_attr1 = interaction[edge_index1[0], edge_index1[1]].view(-1, 1)  # Shape: [num_edges1, 1]

                edge_index2 = (lookahead != 0).nonzero(as_tuple=False).t()  # Shape: [2, num_edges2]
                edge_attr2 = lookahead[edge_index2[0], edge_index2[1]].view(-1, 1)  # Shape: [num_edges2, 1]

                edge_index = torch.cat([edge_index1, edge_index2], dim=1)  # Shape: [2, num_edges1 + num_edges2]
                edge_attr = torch.cat([edge_attr1, edge_attr2], dim=0)


            elif self.gnn_name == 'GCN':
                # Compute edge_index from the union of nonzero entries across all features
                combined_adj_matrix = interaction + lookahead  # Shape: [n_nodes, n_nodes]
                edge_index = (combined_adj_matrix != 0).nonzero(as_tuple=False).t()
                edge_attr = combined_adj_matrix[edge_index[0], edge_index[1]].view(-1, 1)  # Shape: [num_edges1, 1]

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

'''
    def _get_adj_matrix(self, obs) -> torch.Tensor:
        
        Returns the adjacency matrix for the GNN, 
        
        Arguments (no pooling): 
            - obs['interactions']: (batch_size, n_qbits, n_qbits)
            - obs['lookahead']: (batch_size, n_qbits, n_qbits)

        Returns:
            - adj_matrix: (batch_size, 2*n_qbits, n_qbits)
        
        n_qbits = int(obs['n_qbits'][0][0])
        adj_matrix = torch.cat([obs['interactions'][:, :n_qbits, :n_qbits], obs['lookaheads'][:, :n_qbits, :n_qbits]], dim=1)

        if self.device == 'cuda':
            adj_matrix = adj_matrix.cuda()
        return adj_matrix
'''