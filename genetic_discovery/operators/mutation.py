import numpy as np
import random
import networkx as nx

def mutate(n_genes, n_nodes, G):

    mutated_G = G.copy()
    for _ in range(n_genes):
        operation = random.choices(['reverse', 'add', 'delete'], weights=[0.33, 0.33, 0.33])[0]
        edges = list(mutated_G.edges())
        
        if operation == 'delete' and edges: # deletar não gera ciclo
            u, v = random.choice(edges)
            mutated_G.remove_edge(u, v)
            
        elif operation == 'add':
            
            for _ in range(10): 
                u, v = random.sample(range(n_nodes), 2)
                if not mutated_G.has_edge(u, v): 
                    
                    if not nx.has_path(mutated_G, v, u): 
                        mutated_G.add_edge(u, v) # só adiciona se não criar ciclo
                        break 
                        
        elif operation == 'reverse' and edges:
           
            for _ in range(10):
                u, v = random.choice(edges)
                mutated_G.remove_edge(u, v)
                
                if not nx.has_path(mutated_G, u, v):
                    mutated_G.add_edge(v, u)
                    break
                else: # se criar ciclo, volta a aresta original
                   
                    mutated_G.add_edge(u, v)

    return mutated_G

def custom_mutation(offspring, ga_instance):
    n_nodes = int(np.sqrt(offspring.shape[1]))


    for chromosome_idx in range(offspring.shape[0]):
        if random.random() > ga_instance.mutation_probability:
            continue

        max_mutations = max(2, int(0.05 * n_nodes))  # limita o número máximo de mutações para 5% dos nós ou pelo menos 2
        n_genes_to_mutate = random.randint(1, max_mutations)

        matrix = offspring[chromosome_idx].reshape((n_nodes, n_nodes))
        G = nx.DiGraph(matrix) 

        mutated_G = mutate(n_genes_to_mutate, n_nodes, G)

        offspring[chromosome_idx] = nx.to_numpy_array(mutated_G, dtype=int).flatten()
            
    return offspring