import numpy as np
from genetic_discovery.utils.utils import _repair_dag


def crossover(p1, p2):
    """
    Crossover Uniforme de Arestas.
    Para cada par (i, j), o filho herda a conexão de um dos pais aleatoriamente.
    """
    n = p1.shape[0]
    c1 = np.zeros((n, n))
    c2 = np.zeros((n, n))
    
    # Máscara aleatória para decidir de qual pai herdar
    mask = np.random.randint(0, 2, size=(n, n))
    
    # Filho 1 herda de P1 onde mask=1, e de P2 onde mask=0
    c1 = np.where(mask == 1, p1, p2)
    # Filho 2 faz o inverso
    c2 = np.where(mask == 0, p1, p2)
    
    # Garantir que os filhos continuem sendo DAGs
    return _repair_dag(c1), _repair_dag(c2)

