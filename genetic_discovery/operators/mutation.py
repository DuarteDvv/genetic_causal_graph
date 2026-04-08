import numpy as np
import random
from genetic_discovery.utils.utils import _repair_dag

def mutate(adj_matrix, mutation_rate):
    """Mutate an adjacency matrix with a given mutation rate."""
    
    n = adj_matrix.shape[0]
    mutated = adj_matrix.copy()
    
    mutation_type = random.random()
    
    if mutation_type < 0.5:
 
        edges = np.argwhere(mutated == 1)
        if len(edges) > 0:
            i, j = random.choice(edges)
            mutated[i, j] = 0
            mutated[j, i] = 1
    else:
   
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i != j: # Evita self-loops
            mutated[i, j] = 1 - mutated[i, j]
            
    return _repair_dag(mutated)