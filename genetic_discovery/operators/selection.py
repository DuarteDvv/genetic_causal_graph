import random


def select_parents(population, fitness_scores, k=3):
    
    def tournament():
      
        contestants_indices = random.sample(range(len(population)), k) 
        winner_idx = max(contestants_indices, key=lambda i: fitness_scores[i])
        
        return population[winner_idx]
    
    p1 = tournament()
    p2 = tournament()
    
    return p1, p2