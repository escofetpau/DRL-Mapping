import pytest
from gymnasium import spaces
import numpy as np
import torch

from src.models.feature_extractor import GNNFeatureExtractor



@pytest.fixture
def GNN():
    obs_space = spaces.Dict({
        'dummy': spaces.Box(low=0, high=1, shape=(4,), dtype=float)
    })
    return GNNFeatureExtractor(
        observation_space=obs_space,
        action_type='S',
        gnn_name='GATv2',
        device='cpu',
        hidden_features=8,
        n_qbits=4
    )


BATCH_SIZE = 2
N_QBITS = 4
N_CORES = 2

@pytest.fixture
def batch_node_features(): # sergi
    # shape (batch_size, n_qbits, 2 * n_cores + 1)
    features = torch.zeros((BATCH_SIZE, N_QBITS, 2 * N_CORES + 1))
    features[0, 0, N_CORES] = 1
    features[0, 0, -1] = 1
    features[1, 0, N_CORES] = 1
    features[1, 0, -1] = 1

    return features


@pytest.fixture
def batch_adj_matrix():
    lookahead = torch.zeros((BATCH_SIZE, N_QBITS, N_QBITS))
    lookahead[0, 0, 1] = 0.5
    lookahead[0, 1, 0] = 0.5
    lookahead[1, 1, 3] = 0.25
    lookahead[1, 3, 1] = 0.25
    
    interactions = torch.zeros((BATCH_SIZE, N_QBITS, N_QBITS))
    interactions[0, 0, 1] = 1
    interactions[0, 1, 0] = 1
    interactions[1, 2, 3] = 1
    interactions[1, 3, 2] = 1

    adj_matrix = torch.stack([interactions, lookahead], dim=-1) # (batch_size, n_qbits, n_qbits, 2)
    return adj_matrix


def test_batcher(GNN, batch_node_features, batch_adj_matrix):
    '''
        - edge_index: # (2, n_edges)
        - edge_attr: # (n_edges, n_edges_features)
    '''
    batch = GNN.batcher(batch_node_features, batch_adj_matrix)

    N_EDGES = 6

    assert batch.x.shape == (BATCH_SIZE* N_QBITS, 2 * N_CORES + 1)
    assert batch.edge_index.shape == (2, N_EDGES)
    assert batch.edge_attr.shape == (N_EDGES, 2)
    assert torch.equal(batch.batch, torch.arange(BATCH_SIZE).repeat_interleave(N_QBITS))


def test_batcher_vectorized(GNN, batch_node_features, batch_adj_matrix):
    batch = GNN.batcher(batch_node_features, batch_adj_matrix)
    batch_vectorized = GNN.batcher_vectorized(batch_node_features, batch_adj_matrix)
    print(batch)
    print(batch_vectorized)
    assert torch.equal(batch.x, batch_vectorized.x)
    assert torch.equal(batch.edge_index, batch_vectorized.edge_index)
    assert torch.equal(batch.edge_attr, batch_vectorized.edge_attr)
