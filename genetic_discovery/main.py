import pygad
import numpy as np
from genetic_discovery.operators.fitness import build_advanced_fitness
from genetic_discovery.operators.mutation import custom_mutation



def genetic_discovery(data, n_nodes, matrix_initial_pop, num_generations=50, num_parents_mating=5, population_size=20, mutation_rate=0.2):

    flattened_pop = [matrix.flatten() for matrix in matrix_initial_pop]
    
    # preenche o restante da população com matrizes aleatórias
    full_initial_pop = np.random.randint(0, 2, size=(population_size, n_nodes**2)) 
    for i, individual in enumerate(flattened_pop): # 
        full_initial_pop[i] = individual


    custom_fitness = build_advanced_fitness(
        data=data, 
        n_nodes=n_nodes, 
        dag_penalty=1e6,      # Punição letal para ciclos
        penalty_type='AIC'    # Ajuste esse tipo de penalidade se o grafo ficar muito vazio
    )

    ga_instance = pygad.GA(
        initial_population=full_initial_pop, # populacao inicial
        num_generations=num_generations, # n_gerações
        num_parents_mating=num_parents_mating, # n_pais para cruzamento
        fitness_func=custom_fitness, # fitness function

        sol_per_pop=population_size, # soluções (indivíduos) por população (se população for menor que sol_per_pop, o PyGAD irá gerar indivíduos aleatórios para completar)
        num_genes=n_nodes**2, # temos nxn genes (parametros de uma matriz n x n)
        gene_type=int, # estamos otimizando matrizes binarias nxn entao o gene é int
        gene_space=[0, 1], # cada gene pode ser 0 ou 1

        crossover_type="uniform", # tipo de crossover

        mutation_probability=mutation_rate, # taxa de mutação
        mutation_type=custom_mutation, # função de mutação customizada

        parent_selection_type="tournament", # tipo de seleção dos pais
        K_tournament=2, # tamanho do torneio para seleção dos pais

        keep_elitism=1
    )

    ga_instance.run()
    solution, solution_fitness, _ = ga_instance.best_solution()
    best_matrix = solution.reshape((n_nodes, n_nodes))

    return best_matrix