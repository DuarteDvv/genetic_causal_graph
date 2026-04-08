import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BICGauss

def evaluate_candidate(candidate_matrix, data, estimator) -> float:
    """
    candidate_matrix: Matriz de adjacência (Numpy)
    data: O DataFrame com as colunas já nomeadas
    estimator: O objeto BICGauss já inicializado
    """
    nodes = list(data.columns)
    
    # 1. Transformar matriz em lista de arestas (Edges)
    # np.where retorna os índices onde o valor é 1
    idx_from, idx_to = np.where(candidate_matrix == 1)
    edges = [(nodes[i], nodes[j]) for i, j in zip(idx_from, idx_to)]
    
    try:
        # 2. Criar o modelo pgmpy a partir das arestas
        model = BayesianNetwork(edges)
        # Importante: Garantir que todos os nós (mesmo sem arestas) existam no modelo
        model.add_nodes_from(nodes)
        
        # 3. Calcular o score usando o estimador que passamos
        return estimator.score(model)
        
    except Exception as e:
        # Se o GA gerar algo bizarro ou houver erro de ciclo não tratado
        return -1e10 

def evaluate_population(population: list, data) -> list:
    # OTIMIZAÇÃO: Converter para DataFrame e criar o estimador UMA ÚNICA VEZ
    columns = [f"X{i}" for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=columns)
    bic_gauss_estimator = BICGauss(df)
    
    fitness_scores = []
    for candidate in population:
        score = evaluate_candidate(candidate, df, bic_gauss_estimator)
        fitness_scores.append(score)

    return fitness_scores