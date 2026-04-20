import numpy as np
from scipy.linalg import expm

def build_advanced_fitness(data, n_nodes, dag_penalty=1e6, penalty_type='BIC'):
    """
    Fitness function ultrarrápida usando decomposição local e Numpy puro.
    
    penalty_type: 'BIC' (mais esparso, focado em precisão) ou 
                  'AIC' (menos esparso, focado em capacidade preditiva).
    """
    # Garante que os dados sejam um array numpy contíguo
    data_np = data.values if hasattr(data, 'values') else data
    n_samples = data_np.shape[0]

    # --- O CACHE LOCAL ---
    # Armazena o score de um Nó dado seus Pais específicos.
    # Ex: Chave (Alvo: 2, Pais: (0, 3)) -> Score
    local_score_cache = {}

    def get_local_score(target_idx, parent_indices):
        # A tupla de pais garante a imutabilidade para ser chave do dicionário
        parents_tuple = tuple(sorted(parent_indices))
        cache_key = (target_idx, parents_tuple)
        
        if cache_key in local_score_cache:
            return local_score_cache[cache_key]

        y = data_np[:, target_idx]
        
        # Regressão Linear com Mínimos Quadrados (Equações Normais otimizadas)
        if len(parent_indices) == 0:
            # Sem pais: apenas a variância da própria variável (intercepto apenas)
            rss = np.sum((y - np.mean(y))**2)
            k = 1 
        else:
            X = data_np[:, parent_indices]
            # Adiciona coluna de 1s para o intercepto
            X = np.column_stack((np.ones(n_samples), X))
            
            # Resolve a regressão linear OLS: X * beta = y
            # Retorna coeficientes, resíduos, rank, e valores singulares
            _, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
            
            # Captura a Soma dos Quadrados dos Resíduos (RSS)
            if residuals.size > 0:
                rss = residuals[0]
            else:
                rss = np.sum((y - np.mean(y))**2)
                
            k = len(parent_indices) + 1 # Parâmetros: pesos das arestas + intercepto

        # Variância residual (com proteção para evitar log(0))
        var = rss / n_samples
        if var <= 1e-10: 
            var = 1e-10 

        # Log-Likelihood Gaussiana (descartando as constantes aditivas irrelevantes para rank)
        ll = - (n_samples / 2) * np.log(var)

        # Critério de Informação (Queremos MAXIMIZAR esse valor no PyGAD)
        if penalty_type == 'BIC':
            score = ll - (k / 2) * np.log(n_samples)
        elif penalty_type == 'AIC':
            score = ll - k
        else:
            raise ValueError("Use 'BIC' ou 'AIC'")

        # Salva no cache
        local_score_cache[cache_key] = score
        return score


    def fitness_func(ga_instance, solution, solution_idx):
        matrix = solution.reshape((n_nodes, n_nodes))

        # 1. Checagem rígida de ciclo
        h = np.trace(expm(matrix)) - n_nodes
        if h > 1e-5:
            return -dag_penalty * (1 + h)

        # 2. Avaliação Decomposta
        total_score = 0.0
        
        # Para cada nó, olhamos quem são seus pais na matriz de adjacência
        # Usamos <= n_nodes - 1 para iterar pelos índices com segurança
        for target in range(n_nodes):
            # Encontra as linhas (pais) que têm valor 1 na coluna do nó alvo
            parents = np.where(matrix[:, target] == 1)[0]
            
            # Soma o score local
            total_score += get_local_score(target, parents)

        return total_score

    return fitness_func