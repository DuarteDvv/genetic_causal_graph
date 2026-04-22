import numpy as np
import random

def custom_mutation(offspring, ga_instance):
    """
    Aplica Adição, Remoção ou Inversão de arestas de forma aleatória.
    """
    # numero de nós = raiz quadrada do número de genes (que é n^2 para matrizes n x n)
    n_nodes = int(np.sqrt(offspring.shape[1]))
    
    for chromosome_idx in range(offspring.shape[0]):

        if random.random() > ga_instance.mutation_probability:
            continue
            
        matrix = offspring[chromosome_idx].reshape((n_nodes, n_nodes))
        
        # sortear operacao de mutacao: 40% Inverter (explora MEC), 30% Adicionar, 30% Deletar
        operation = random.choices(['reverse', 'add', 'delete'], weights=[0.4, 0.3, 0.3])[0]
        
        # pega as posições das arestas existentes
        existing_edges = np.argwhere(matrix == 1)
        
        if operation == 'reverse' and len(existing_edges) > 0:

            idx = random.randint(0, len(existing_edges) - 1)
            row, col = existing_edges[idx]
            matrix[row, col] = 0
            matrix[col, row] = 1
            
        elif operation == 'delete' and len(existing_edges) > 0:

            idx = random.randint(0, len(existing_edges) - 1)
            row, col = existing_edges[idx]
            matrix[row, col] = 0
            
        elif operation == 'add':
            
            non_edges = np.argwhere(matrix == 0)
            valid_non_edges = [pos for pos in non_edges if pos[0] != pos[1]]
            
            if valid_non_edges:
                idx = random.randint(0, len(valid_non_edges) - 1)
                row, col = valid_non_edges[idx]

                # verifica se a adição da aresta criaria um ciclo (ou seja, se já existe uma aresta oposta)
                if matrix[col, row] == 0:
                    matrix[row, col] = 1
                    
        offspring[chromosome_idx] = matrix.flatten()
        
    return offspring