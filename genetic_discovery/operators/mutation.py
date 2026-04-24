import numpy as np
import random
import networkx as nx

def custom_mutation(offspring, ga_instance):
    n_nodes = int(np.sqrt(offspring.shape[1]))

    genes_proportion = 0.05
    num_genes_to_mutate = max(1, int(genes_proportion * offspring.shape[1])) # testar depois

    for chromosome_idx in range(offspring.shape[0]):
        if random.random() > ga_instance.mutation_probability:
            continue
            
        matrix = offspring[chromosome_idx].reshape((n_nodes, n_nodes))
        G = nx.DiGraph(matrix) 
        
        operation = random.choices(['reverse', 'add', 'delete'], weights=[0.33, 0.33, 0.33])[0]
        edges = list(G.edges())
        
        if operation == 'delete' and edges: # deletar não gera ciclo
            u, v = random.choice(edges)
            G.remove_edge(u, v)
            
        elif operation == 'add':
            
            for _ in range(10): 
                u, v = random.sample(range(n_nodes), 2)
                if not G.has_edge(u, v): 
                    
                    if not nx.has_path(G, v, u): 
                        G.add_edge(u, v) # só adiciona se não criar ciclo
                        break 
                        
        elif operation == 'reverse' and edges:
           
            for _ in range(10):
                u, v = random.choice(edges)
                G.remove_edge(u, v)
                
                if not nx.has_path(G, u, v):
                    G.add_edge(v, u)
                    break
                else: # se criar ciclo, volta a aresta original
                   
                    G.add_edge(u, v)

        offspring[chromosome_idx] = nx.to_numpy_array(G, dtype=int).flatten()
            
    return offspring