
import random
from genetic_discovery.operators.fitness import evaluate_population
from genetic_discovery.operators.selection import select_parents
from genetic_discovery.operators.crossover import crossover
from genetic_discovery.operators.mutation import mutate


def genetic_discovery(
        data,
        max_generations: int, n_population: int, 
        mutation_rate: float, n_childrens: int,
        crossover_rate: float, initial_population: list
        ):
    
    """"
    Run the genetic algorithm for causal discovery

    input: 
    - data: observed data
    - max_generations: number of iterations of the genetic algorithm
    - n_population: number of candidates selected in each generation
    - mutation_rate: probability of mutating a candidate solution
    - crossover_rate: probability of crossing over two candidate solutions
    - initial_population: initial set of adjacency matrices representing candidate causal graphs
    - n_childrens: number of children generated in each generation

    output:

    - best_candidate: the best candidate solution found after max_generations
  
    """

    population = list(initial_population)
    while len(population) < n_population:
      
        base = random.choice(initial_population)
        population.append(mutate(base.copy(), mutation_rate=0.5))

    for _ in range(max_generations):

        childrens = []
        while len(childrens) < n_childrens:
            p1, p2 = select_parents(population, evaluate_population(population, data), k=3)

            if random.random() < crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1, p2

            if random.random() < mutation_rate:
                c1 = mutate(c1, mutation_rate)

            if random.random() < mutation_rate:
                c2 = mutate(c2, mutation_rate)

            if len(childrens) < n_childrens:
                childrens.append(c1)
            if len(childrens) < n_childrens:
                childrens.append(c2)
        
        all_candidates = population + childrens
        fitness_scores = evaluate_population(all_candidates, data)

        ranked = sorted(zip(all_candidates, fitness_scores), key=lambda x: x[1], reverse=True)

        new_population = []
        seen_hashes = set()
        for cand, _ in ranked:
            h = hash(cand.tobytes())
            if h not in seen_hashes:
                new_population.append(cand)
                seen_hashes.add(h)
                
            if len(new_population) == n_population:
                break

        population = new_population
            

    return population[0]