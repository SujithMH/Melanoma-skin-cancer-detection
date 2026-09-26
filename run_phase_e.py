import argparse
import os
from src.hpo.runner import HPORunner
from src.hpo.random_search import run_random_search
from src.hpo.gwo import run_gwo
from src.hpo.ga import run_ga

def main():
    parser = argparse.ArgumentParser(description="Run HPO for Melanoma Hybrid Net")
    parser.add_argument('--algo', type=str, required=True, choices=['rs', 'gwo', 'ga'], help="Algorithm to run")
    parser.add_argument('--seed', type=int, required=True, help="Random seed (e.g., 42, 101, 2024)")
    args = parser.parse_args()
    
    os.makedirs('logs', exist_ok=True)
    log_file = f"logs/hpo_{args.algo}_seed{args.seed}.jsonl"
    
    runner = HPORunner(log_file=log_file, seed=args.seed)
    
    # Budget: 60 evaluations per run (Pop size 6 x 10 iterations)
    # At ~7 mins per eval, this is ~7 hours per run.
    budget = 60 
    
    if args.algo == 'rs':
        best_vec, best_score = run_random_search(runner, num_evals=budget)
    elif args.algo == 'gwo':
        best_vec, best_score = run_gwo(runner, num_wolves=6, max_iter=10)
    elif args.algo == 'ga':
        best_vec, best_score = run_ga(runner, pop_size=6, max_gen=10)
        
    print(f"\nFINISHED {args.algo.upper()} (Seed {args.seed})")
    print(f"Best Score: {best_score:.4f}")

if __name__ == "__main__":
    main()