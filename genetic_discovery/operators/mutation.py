import numpy as np
import random

def custom_mutation(offspring, ga_instance):
    """
    Mutação customizada para matrizes de adjacência (DAGs).
    Aplica Adição, Remoção ou Inversão de arestas de forma estruturada.
    """
    # Acessamos o número de nós tirando a raiz quadrada do tamanho do cromossomo
    n_nodes = int(np.sqrt(offspring.shape[1]))
    
    for chromosome_idx in range(offspring.shape[0]):
        # A mutação só ocorre de acordo com a probabilidade definida no GA
        if random.random() > ga_instance.mutation_probability:
            continue
            
        # Reconstrói a matriz
        matrix = offspring[chromosome_idx].reshape((n_nodes, n_nodes))
        
        # Sorteia qual operação estrutural fazer
        # 40% Inverter (explora MEC), 30% Adicionar, 30% Deletar
        operation = random.choices(['reverse', 'add', 'delete'], weights=[0.4, 0.3, 0.3])[0]
        
        # Pega as coordenadas das arestas existentes e inexistentes
        existing_edges = np.argwhere(matrix == 1)
        
        if operation == 'reverse' and len(existing_edges) > 0:
            # Escolhe uma aresta aleatória e inverte
            idx = random.randint(0, len(existing_edges) - 1)
            row, col = existing_edges[idx]
            matrix[row, col] = 0
            matrix[col, row] = 1
            
        elif operation == 'delete' and len(existing_edges) > 0:
            # Apaga uma aresta
            idx = random.randint(0, len(existing_edges) - 1)
            row, col = existing_edges[idx]
            matrix[row, col] = 0
            
        elif operation == 'add':
            # Acha onde não tem aresta (ignorando a diagonal principal)
            non_edges = np.argwhere(matrix == 0)
            valid_non_edges = [pos for pos in non_edges if pos[0] != pos[1]]
            
            if valid_non_edges:
                idx = random.randint(0, len(valid_non_edges) - 1)
                row, col = valid_non_edges[idx]
                # Verifica se o inverso já não existe para não criar um ciclo de 2 nós instantâneo
                if matrix[col, row] == 0:
                    matrix[row, col] = 1
                    
        # Achata a matriz de volta para o PyGAD
        offspring[chromosome_idx] = matrix.flatten()
        
    return offspring