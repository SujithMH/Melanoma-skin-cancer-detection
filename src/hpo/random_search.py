
import numpy as np

def run_random_search(runner, num_evals):
    print("\nStarting Random Search...")
    np.random.seed(runner.seed)
    
    # Check how many evals we already did
    start_idx = len(runner.history)
    
    for i in range(start_idx, num_evals):
        print(f"Random Search Eval {i+1}/{num_evals}")
        vector = np.random.rand(7)
        runner.evaluate(vector)
        
    return runner.best_vec, runner.best_score