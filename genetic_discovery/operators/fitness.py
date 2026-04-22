import numpy as np
from scipy.linalg import expm

class DagFitness:
    """
    Callable picklable para uso com multiprocessing no PyGAD.
    """

    def __init__(self, data, n_nodes, dag_penalty=1e6, penalty_type='BIC'):
        self.data_np = data.values if hasattr(data, 'values') else data
        self.n_nodes = n_nodes
        self.dag_penalty = dag_penalty
        self.penalty_type = penalty_type
        self.n_samples = self.data_np.shape[0]
        self.local_score_cache = {}

    def _get_local_score(self, target_idx, parent_indices):
        parents_tuple = tuple(sorted(parent_indices))
        cache_key = (target_idx, parents_tuple)

        if cache_key in self.local_score_cache:
            return self.local_score_cache[cache_key]

        y = self.data_np[:, target_idx]

        # Regressao linear via minimos quadrados.
        if len(parent_indices) == 0:
            rss = np.sum((y - np.mean(y)) ** 2)
            k = 1
        else:
            X = self.data_np[:, parent_indices]
            X = np.column_stack((np.ones(self.n_samples), X))

            _, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
            if residuals.size > 0:
                rss = residuals[0]
            else:
                rss = np.sum((y - np.mean(y)) ** 2)

            k = len(parent_indices) + 1

        var = rss / self.n_samples
        if var <= 1e-10:
            var = 1e-10

        ll = -(self.n_samples / 2) * np.log(var)

        if self.penalty_type == 'BIC':
            score = ll - (k / 2) * np.log(self.n_samples)
        elif self.penalty_type == 'AIC':
            score = ll - k
        else:
            raise ValueError("Use 'BIC' ou 'AIC'")

        self.local_score_cache[cache_key] = score
        return score

    def __call__(self, ga_instance, solution, solution_idx):
        matrix = solution.reshape((self.n_nodes, self.n_nodes))

        # Penalidade de ciclos: h = trace(expm(A)) - n_nodes.
        h = np.trace(expm(matrix)) - self.n_nodes
        if h > 1e-5:
            return -self.dag_penalty * (1 + h)

        total_score = 0.0
        for target in range(self.n_nodes):
            parents = np.where(matrix[:, target] == 1)[0]
            total_score += self._get_local_score(target, parents)

        return total_score


def build_fitness(data, n_nodes, dag_penalty=1e6, penalty_type='BIC'):
    """
    Fitness function para otimizacao de grafos causais usando BIC/AIC e penalidade para ciclos.
    """
    return DagFitness(
        data=data,
        n_nodes=n_nodes,
        dag_penalty=dag_penalty,
        penalty_type=penalty_type,
    )