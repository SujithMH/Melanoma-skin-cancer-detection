import numpy as np

def run_ga(runner, pop_size=6, max_gen=10):
    print("\nStarting Genetic Algorithm (GA)...")
    np.random.seed(runner.seed)
    
    pop = np.random.rand(pop_size, 7)
    start_gen = len(runner.history) // pop_size
    
    for gen in range(start_gen, max_gen):
        print(f"\n--- GA Generation {gen+1}/{max_gen} ---")
        
        # Evaluate
        scores = np.array([runner.evaluate(ind) for ind in pop])
        
        # Elitism: Keep the best individual
        best_idx = np.argmax(scores)
        next_pop = [pop[best_idx].copy()]
        
        # Tournament Selection & Crossover
        while len(next_pop) < pop_size:
            # Select 2 parents via tournament (size 2)
            parents = []
            for _ in range(2):
                idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
                winner = idx1 if scores[idx1] > scores[idx2] else idx2
                parents.append(pop[winner])
                
            # Blend Crossover (alpha = 0.5)
            alpha = np.random.rand(7)
            child = alpha * parents[0] + (1 - alpha) * parents[1]
            
            # Gaussian Mutation (10% chance per gene)
            for j in range(7):
                if np.random.rand() < 0.1:
                    child[j] += np.random.normal(0, 0.1)
                    
            child = np.clip(child, 0.0, 1.0)
            next_pop.append(child)
            
        pop = np.array(next_pop)
        
    best_idx = np.argmax(scores)
    return pop[best_idx], scores[best_idx]