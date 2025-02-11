from pydantic import _getattr_migration
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Data, Batch
from src.models.gatv2 import GATv2Model
from torch_geometric.nn.models import GCN
from src.utils.constants import N_CORES, HIDDEN_FEATURES_GAT, NUM_HEADS, HIDDEN_LAYERS_GAT, GAT_DROPOUT, OUT_FEATURES_GAT, ST_OUTPUT_DIM, ST_HIDDEN_DIM
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GATv2FeatureExtractor(BaseFeaturesExtractor):
    '''
    El feature extractor s'encarrega de redimensionar les observacions i fer el forward per la GAT.
    '''
    def __init__(self, observation_space, in_features=2 * N_CORES + 1, edge_dim=2, 
                 hidden_features_gat=HIDDEN_FEATURES_GAT, out_features_gat=OUT_FEATURES_GAT,
                 num_heads=NUM_HEADS, hidden_layers_gat=HIDDEN_LAYERS_GAT, gat_dropout=GAT_DROPOUT):
        super(GATv2FeatureExtractor, self).__init__(observation_space, 
                                                   features_dim=out_features_gat * 8 + N_CORES)
        

        #self.gat = GATv2Model(in_features, edge_dim, hidden_features_gat, out_features_gat, num_heads, hidden_layers_gat, gat_dropout)
        self.gat = GCN(in_features, hidden_features_gat, hidden_layers_gat, out_features_gat, gat_dropout)
        self.gat.to(device)
    

    def forward(self, obs):
        #Durant l'exploració és 1, durant l'entrenament es BATCH_SIZE
        batch_size = obs['node_features'].shape[0]
        #El retall s'haurà de fer dins del batcher una vegada hi hagi diversos nombres de qbits dins d'un batch
        #Ara agafo n_qbits del primer circuit però això no serà sempre veritat. TODO
        n_qbits = int(obs['n_qbits'][0][0])
        x = obs['node_features'][:, :n_qbits, : n_qbits]
        adj_mat = obs['adj_matrix'][: ,:n_qbits, : n_qbits]

        #Passo pel batcher per a construir un sol estat no connex que contingui tots els estats
        batch = self.batcher(batch_size, x, adj_mat)

        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_features = batch.edge_attr.to(device)
        edge_features = edge_features.sum(axis=1) # combine interaction and lookahead for GCN

        node_embeddings = self.gat(x, edge_index, edge_features)
        #De nou, quan hi hagi diversos nombres de qbits, s'haurà de refer utilitzant el batch.batch, que indica a quina observació
        #Pertany cada node TODO
        

        node_embeddings = node_embeddings.reshape(batch_size, n_qbits*OUT_FEATURES_GAT) #Batched flatten
        node_embeddings = torch.cat((node_embeddings, obs['core_capacities']), dim = 1)
        return node_embeddings
        '''
        Ara s'estan aplanant les node_embeddings, amb el set transformer el procés canviarà
        '''
        node_embeddings = node_embeddings.unsqueeze(0)

        #El que abans era un graf tot junt ara es torna a separar per instàncies dins del batch
        node_embeddings = node_embeddings.view(batch_size, n_qbits, OUT_FEATURES_GAT)

        return node_embeddings


    def batcher(self, batch_size, batch_node_features, batch_adj_matrix):
        '''
        Constructs a PyG.Data object for each graph observation and combines them into a batch.
        Handles multiple edge attributes (e.g., multiple features per edge).
        '''
        data_list = []
        for i in range(batch_size):
            # Node features for the i-th graph
            node_features = batch_node_features[i]  # Shape: [n_nodes, node_feature_dim]

            # Adjacency matrix for the i-th graph, with multiple edge attributes
            adj_matrix = batch_adj_matrix[i]  # Shape: [num_edge_features, n_nodes, n_nodes]

            # Compute edge_index from the union of nonzero entries across all features
            combined_adj_matrix = adj_matrix.sum(dim=0)  # Shape: [n_nodes, n_nodes]
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
