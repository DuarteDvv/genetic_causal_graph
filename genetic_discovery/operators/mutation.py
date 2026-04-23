import numpy as np
import random
import networkx as nx

def custom_mutation(offspring, ga_instance):

    reverse_prob = 0.2
    add_prob = 1 - reverse_prob / 4
    delete_prob = 321 - reverse_prob / 4

    n_nodes = int(np.sqrt(offspring.shape[1]))
    
    for chromosome_idx in range(offspring.shape[0]):
        if random.random() > ga_instance.mutation_probability:
            continue
            
        matrix = offspring[chromosome_idx].reshape((n_nodes, n_nodes))
        operation = random.choices(['reverse', 'add', 'delete'], weights=[0.3, 0.4, 0.3])[0]
        existing_edges = np.argwhere(matrix == 1)
        
        # Fazemos uma copia provisoria para testar ciclos
        temp_matrix = matrix.copy()
        
        if operation == 'reverse' and len(existing_edges) > 0:
            idx = random.randint(0, len(existing_edges) - 1)
            row, col = existing_edges[idx]
            temp_matrix[row, col] = 0
            temp_matrix[col, row] = 1
            
        elif operation == 'delete' and len(existing_edges) > 0:
            idx = random.randint(0, len(existing_edges) - 1)
            row, col = existing_edges[idx]
            temp_matrix[row, col] = 0
            
        elif operation == 'add':
            non_edges = np.argwhere(matrix == 0)
            valid_non_edges = [pos for pos in non_edges if pos[0] != pos[1]]
            if valid_non_edges:
                idx = random.randint(0, len(valid_non_edges) - 1)
                row, col = valid_non_edges[idx]
                temp_matrix[row, col] = 1
        
        # Teste de validade essencial: só aplica se mantiver um DAG
        if nx.is_directed_acyclic_graph(nx.DiGraph(temp_matrix)):
            offspring[chromosome_idx] = temp_matrix.flatten()
            
    return offspring