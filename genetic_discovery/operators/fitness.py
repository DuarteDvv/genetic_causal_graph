import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BICGauss
from scipy.linalg import expm


def acyclicity_penalty(adj_matrix):
    d = adj_matrix.shape[0] # n of nodes
    A = adj_matrix # bin matrix

    h = np.trace(expm(A)) - d
    return h # h = 0 if DAG, h > 0 if has cycles


def bicg_score(candidate_matrix, data, estimator) -> float:
    """
    candidate_matrix: Matriz de adjacência (Numpy)
    data: O DataFrame com as colunas já nomeadas
    estimator: O objeto BICGauss já inicializado
    """
    nodes = list(data.columns)
    
    idx_from, idx_to = np.where(candidate_matrix == 1)
    edges = [(nodes[i], nodes[j]) for i, j in zip(idx_from, idx_to)]
    
    try:
        model = BayesianNetwork(edges)
        model.add_nodes_from(nodes)
        
        return estimator.score(model)
        
    except Exception as e:
      
        return -1e10 

def evaluate_population(population: list, data, dag_penalty: float = 1.0) -> list:
  
    columns = [f"X{i}" for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=columns)
    bic_gauss_estimator = BICGauss(df)
    
    fitness_scores = []
    for candidate in population:
        h = acyclicity_penalty(candidate)
        gauss_bic_score = bicg_score(candidate, df, bic_gauss_estimator)
        fitness_scores.append(gauss_bic_score - dag_penalty * h)  

    return fitness_scores


