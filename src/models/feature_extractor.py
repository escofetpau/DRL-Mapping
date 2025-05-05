import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Batch, Data
from torch_geometric.nn.models import GCN, GAT


class GNNFeatureExtractor(BaseFeaturesExtractor):
    '''
    El feature extractor s'encarrega de redimensionar les observacions i fer el forward per la GNN.
    '''
    def __init__(self, observation_space, action_type: str, gnn_name: str, device: str,
                 hidden_features, n_qubits: int, n_cores: int, out_features: int = 16,
                 num_heads: int = 1, hidden_layers: int = 4, dropout: float=0):
        assert gnn_name in ['GATv2', 'GCN'], f'Invalid GNN model name: {gnn_name}'

        super().__init__(observation_space, features_dim=out_features * n_qubits + n_cores)
        
        self.device = device
        self.action_type = action_type
        self.out_features = out_features
        self.gnn_name = gnn_name
        self.n_cores = n_cores

        in_features = 2*self.n_cores + 1 if self.action_type == 'S' else 2*self.n_cores

        if self.gnn_name == 'GATv2':
            self.gnn = GAT(in_features, hidden_features, hidden_layers, out_features, dropout, v2 = True, edge_dim = 2)

        else:
            self.gnn = GCN(in_features, hidden_features, hidden_layers, out_features, dropout)
    
        self.gnn.to(self.device)


    def _get_batch_node_features(self, obs) -> torch.Tensor:
        '''
        Returns the node features for the GNN, which are the new and old allocations.

        Arguments:
            - obs['new_allocation']: torch.Tensor (batch_size, n_qubits, n_cores)
            - obs['old_allocation']: torch.Tensor (batch_size, n_qubits, n_cores)
            - obs['flag']: (batch_size, n_qubits, 1) if action = 'S'

        Returns:
            - node_features: (batch_size, n_qubits, 2*n_cores + 1) / (batch_size, n_qubits, 2*n_cores)
        '''
        n_qubits = int(obs['n_qubits'][0][0])
        features = torch.cat([obs['new_allocation'][:, :n_qubits, :n_qubits], obs['old_allocation'][:, :n_qubits, :n_qubits]], dim=2)

        if self.action_type == 'S':
            features = torch.cat([features, obs['flag']], dim=2)

        if self.device == 'cuda':
            features = features.cuda()

        return features

    def _get_batch_adj_matrix(self, obs) -> torch.Tensor:
        '''
        Returns the adjacency matrix for the GNN
        
        Arguments (no pooling): 
            - obs['interactions']: (batch_size, n_qubits, n_qubits)
            - obs['lookahead']: (batch_size, n_qubits, n_qubits)

        Returns:
            - adj_matrix: (batch_size, n_qubits, n_qubits, 2)
        '''

        n_qubits = int(obs['n_qubits'][0][0])
        adj_matrix = torch.stack([obs['interactions'][:, :n_qubits, :n_qubits], obs['lookaheads'][:, :n_qubits, :n_qubits]], dim=-1)

        if self.device == 'cuda':
            adj_matrix = adj_matrix.cuda()
        return adj_matrix


    def forward(self, obs):
        batch_size = obs['new_allocation'].shape[0]
        n_qubits = int(obs['n_qubits'][0][0])
        x = self._get_batch_node_features(obs)
        adj_matrix = self._get_batch_adj_matrix(obs)

        batch = self.batcher_vectorized(x, adj_matrix)

        x = batch.x.to(self.device)
        edge_index = batch.edge_index.to(self.device) # (2, n_edges)
        edge_attr = batch.edge_attr.to(self.device) # (n_edges, n_edges_features)
 
        node_embeddings = self.gnn(x, edge_index, edge_attr)

        node_embeddings = node_embeddings.reshape(batch_size, n_qubits*self.out_features) # Batched flatten
        node_embeddings = torch.cat((node_embeddings, obs['core_capacities']), dim = 1)
        return node_embeddings


    def batcher_vectorized(self, batch_node_features: torch.Tensor, batch_adj_matrix: torch.Tensor) -> Batch:
        '''
        Constructs a PyG.Batch object from a batch of graphs with two edge types (interaction & lookahead).
        Args:
            - batch_node_features: (batch_size, n_qubits, 2*n_cores + 1) / (batch_size, n_qubits, 2*n_cores)
            - batch_adj_matrix: (batch_size, n_qubits, n_qubits, 2)
        '''
        batch_size, n_qubits, node_feature_size = batch_node_features.shape
        device = batch_node_features.device

        # Nodes
        x = batch_node_features.reshape(-1, node_feature_size)  # (batch_size * n_qubits, node_feature_size)
        node_batch = torch.arange(batch_size, device=device).repeat_interleave(n_qubits)    # [0..0, 1..1, ... batch_size-1..batch_size-1]

        # Edges
        edge_mask = (batch_adj_matrix.sum(dim=-1) != 0)  # (batch_size, n_qubits, n_qubits)
        batch_idx, src, dst = edge_mask.nonzero(as_tuple=True) # (n_edges), (n_edges), (n_edges)
        offset = batch_idx * n_qubits   # to match with node_batch

        edge_index = torch.stack([src + offset, dst + offset], dim=0)  # (2, n_edges)
        edge_attr = batch_adj_matrix[batch_idx, src, dst]  # (n_edges, 2)

        if self.gnn_name == 'GCN':
            edge_attr = edge_attr.sum(dim=-1, keepdim=True) # (n_edges, 1)

        batch = Batch(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=node_batch
        )

        return batch


    def batcher(self, batch_node_features: torch.Tensor, batch_adj_matrix: torch.Tensor) -> Batch:
        '''
        Constructs a PyG.Data object for each graph observation and combines them into a batch.
        Handles multiple edge attributes (e.g., multiple features per edge).
        '''
        data_list = []
        batch_size = batch_node_features.shape[0]

        for i in range(batch_size):
            node_features = batch_node_features[i]  # [n_qubits, 2*n_cores (+1)]
            adj_matrix = batch_adj_matrix[i]  # [n_qubits, n_qubits, 2]

            edge_index = (adj_matrix.sum(-1) != 0).nonzero(as_tuple=False).t() # [2, num_edges1]
            edge_attr = adj_matrix[edge_index[0], edge_index[1]]  # [num_edges, n_features]

            if self.gnn_name == 'GCN':
                edge_attr = edge_attr.sum(dim=-1, keepdim=True)

            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr
            )
            data_list.append(data)

        batch = Batch.from_data_list(data_list)
        return batch
