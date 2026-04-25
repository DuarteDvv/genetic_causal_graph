import pygad
import numpy as np
from genetic_discovery.operators.fitness import build_fitness
from genetic_discovery.operators.crossing_over import custom_crossing_over
from genetic_discovery.operators.mutation import custom_mutation

def on_gen(ga_instance):
    print(f"Generation {ga_instance.generations_completed}: Best Fitness = {ga_instance.best_solution()[1]}")

def genetic_discovery(data, n_nodes, matrix_initial_pop, num_generations=50, num_parents_mating=5, population_size=20, mutation_rate=0.2):

    flattened_pop = [matrix.flatten() for matrix in matrix_initial_pop]
    
    full_initial_pop = []
    for _ in range(population_size):

        # matriz aleatória apenas no triângulo inferior para evitar ciclos
        mat = np.tril(np.random.choice([0, 1], size=(n_nodes, n_nodes), p=[0.8, 0.2]), -1)
        p = np.random.permutation(n_nodes)
        mat = mat[p][:, p]
        full_initial_pop.append(mat.flatten())

    full_initial_pop = np.array(full_initial_pop)


    for i, individual in enumerate(flattened_pop):
        full_initial_pop[i] = individual

    #normized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)


    custom_fitness = build_fitness(
        data= data, 
        n_nodes=n_nodes, 
        penalty_type='BIC'    # "BIC" ou "AIC"
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

        crossover_type=custom_crossing_over, # crossover por no

        mutation_probability=mutation_rate, # taxa de mutação
        mutation_type=custom_mutation, # função de mutação 

        parent_selection_type="tournament", # tipo de seleção dos pais
        K_tournament=2, # tamanho do torneio para seleção dos pais

        keep_elitism=2,

        #parallel_processing=["thread",8], # threads
        on_generation=on_gen, 
        random_seed=42,
        stop_criteria=["saturate_50"] # para se a melhor solução não melhorar por 50 gerações seguidas
    )

    ga_instance.run()
    solution, solution_fitness, _ = ga_instance.best_solution()
    best_matrix = solution.reshape((n_nodes, n_nodes))

    return best_matrix