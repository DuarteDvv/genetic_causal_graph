import numpy as np
import random
import networkx as nx


def custom_crossover(parents, offspring_size, ga_instance):
    n_nodes = int(np.sqrt(parents.shape[1]))
    offspring = np.empty(offspring_size, dtype=int)

    for k in range(offspring_size[0]):
        parent1 = parents[k % parents.shape[0]].reshape((n_nodes, n_nodes))
        parent2 = parents[(k + 1) % parents.shape[0]].reshape((n_nodes, n_nodes))

        # onde os dois pais concordam 
        child_matrix = np.logical_and(parent1, parent2).astype(int)
        G_child = nx.DiGraph(child_matrix)

        # arestas que estão no Pai 1 OU no Pai 2, mas não em ambos
        disputed_edges_p1 = [e for e in zip(*np.where(parent1 == 1)) if child_matrix[e] == 0]
        disputed_edges_p2 = [e for e in zip(*np.where(parent2 == 1)) if child_matrix[e] == 0]
        
        # junta todas 
        all_disputed = disputed_edges_p1 + disputed_edges_p2
        random.shuffle(all_disputed)

        # tenta adicionar as arestas disputadas se não formarem ciclo
        for u, v in all_disputed:
            # 50% de chance de herdar a aresta, mas só se não criar ciclo
            if random.random() < 0.5:
                
                if not nx.has_path(G_child, v, u):
                    G_child.add_edge(u, v)

        offspring[k, :] = nx.to_numpy_array(G_child, dtype=int).flatten()

    return offspring