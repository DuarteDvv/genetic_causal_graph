import networkx as nx
import random

def _repair_dag(adj_matrix):
    """
    Função auxiliar para garantir que a matriz de adjacência seja um DAG.
    Se houver um ciclo, remove a aresta que o fecha.
    """
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    while not nx.is_directed_acyclic_graph(G):
        # Encontra um ciclo e remove uma aresta aleatória dele
        try:
            cycle = nx.find_cycle(G)
            u, v = random.choice(cycle)
            G.remove_edge(u, v)
        except nx.NetworkXNoCycle:
            break
    return nx.to_numpy_array(G)