import numpy as np
import random
import networkx as nx


def _repair_dag(matrix):
    """Remove arestas ate obter um DAG valido."""
    graph = nx.DiGraph(matrix)

    while not nx.is_directed_acyclic_graph(graph):
        cycle_edges = list(nx.find_cycle(graph, orientation="original"))
        edge_to_remove = random.choice(cycle_edges)
        graph.remove_edge(edge_to_remove[0], edge_to_remove[1])

    repaired = nx.to_numpy_array(graph, dtype=int)
    np.fill_diagonal(repaired, 0)
    return repaired

def custom_crossover(parents, offspring_size, ga_instance):
    """
    Crossover por no: para cada coluna (alvo), o filho herda todos os pais
    de um dos dois genitores.
    """
    n_nodes = int(np.sqrt(parents.shape[1]))
    offspring = np.empty(offspring_size, dtype=int)

    for k in range(offspring_size[0]):
        parent1 = parents[k % parents.shape[0]].reshape((n_nodes, n_nodes))
        parent2 = parents[(k + 1) % parents.shape[0]].reshape((n_nodes, n_nodes))

        child = np.zeros((n_nodes, n_nodes), dtype=int)

        for target_node in range(n_nodes):
            if random.random() < 0.5:
                child[:, target_node] = parent1[:, target_node]
            else:
                child[:, target_node] = parent2[:, target_node]

        np.fill_diagonal(child, 0)
        child = _repair_dag(child)
        offspring[k, :] = child.flatten()

    return offspring
    

