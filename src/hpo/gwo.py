import numpy as np

def run_gwo(runner, num_wolves=6, max_iter=10):
    print("\nStarting Grey Wolf Optimizer (GWO)...")
    np.random.seed(runner.seed)
    
    # Initialize population
    positions = np.random.rand(num_wolves, 7)
    
    # Check for resume state
    evals_done = len(runner.history)
    start_iter = evals_done // num_wolves
    
    # If we are exactly in the middle of a generation due to a crash, 
    # we just re-evaluate the current generation (runner caches results anyway)
    
    alpha_pos, alpha_score = np.zeros(7), 0.0
    beta_pos, beta_score = np.zeros(7), 0.0
    delta_pos, delta_score = np.zeros(7), 0.0
    
    for t in range(start_iter, max_iter):
        print(f"\n--- GWO Iteration {t+1}/{max_iter} ---")
        
        # Evaluate population
        scores = []
        for i in range(num_wolves):
            score = runner.evaluate(positions[i])
            scores.append(score)
            
            # Update Alpha, Beta, Delta (Maximization problem)
            if score > alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = score, positions[i].copy()
            elif score > beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = score, positions[i].copy()
            elif score > delta_score:
                delta_score, delta_pos = score, positions[i].copy()
                
        # Update positions
        a = 2.0 - t * (2.0 / max_iter) # Decreases linearly from 2 to 0
        
        for i in range(num_wolves):
            for j in range(7):
                # Alpha math
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha
                
                # Beta math
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta
                
                # Delta math
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta
                
                # Average and clip to [0, 1] bounds
                new_pos = (X1 + X2 + X3) / 3.0
                positions[i, j] = np.clip(new_pos, 0.0, 1.0)
                
    return alpha_pos, alpha_score