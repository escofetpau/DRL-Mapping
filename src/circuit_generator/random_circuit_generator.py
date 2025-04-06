import numpy as np
import sparse
import random

def generate_circuit(num_slices, num_qubits, gates_per_slice):
    # Function to generate a single sparse matrix with the given constraints
    def generate_sparse_matrix(num_qubits, num_gates):
        rows = []
        cols = []
        nodes = list(range(num_qubits))

        # Shuffle nodes to ensure randomness
        np.random.shuffle(nodes)

        for i in range(num_gates):
            # Take two nodes at a time
            node_a, node_b = nodes.pop(), nodes.pop()

            # Add the connection (twice, as per the adjacency matrix requirement)
            rows.extend([node_a, node_b])
            cols.extend([node_b, node_a])

        # Create the sparse matrix in COO format
        data = [1] * len(rows)
        return rows, cols, data

    # Generate the sparse matrix data for all matrices
    all_coords = [[], [], []]  # To store (matrix_index, row, col)
    all_data = []
    for idx in range(num_slices):
        num_gates = random.choice(gates_per_slice)  # Randomly select the number of gates for this slice
        rows, cols, data = generate_sparse_matrix(num_qubits, num_gates)
        all_coords[0].extend([idx] * len(rows))  # Add the matrix index as the first dimension
        all_coords[1].extend(rows)
        all_coords[2].extend(cols)
        all_data.extend(data)

    # Create the 3D sparse matrix in COO format
    sparse_tensor = sparse.COO(coords=all_coords, data=all_data, shape=(num_slices, num_qubits, num_qubits))
    return sparse_tensor
