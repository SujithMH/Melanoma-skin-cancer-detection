import numpy as np
import time
from src.hpo.encoding import decode_vector
from src.hpo.objective import evaluate_candidate

def main():
    print("==================================================")
    print("PHASE D — NOISE STUDY & HARNESS MEASUREMENT")
    print("==================================================")
    
    np.random.seed(100)
    # Generate 3 random configurations in [0, 1]^7
    configs = [np.random.rand(7) for _ in range(3)]
    seeds = [42, 101, 2024]
    
    results = {}
    
    for cfg_idx, vec in enumerate(configs):
        cfg_dict = decode_vector(vec)
        print(f"\nEvaluating Config #{cfg_idx + 1}:")
        print(f"  Fusion: {cfg_dict['fusion_type']} | FC Hidden: {cfg_dict['fc_hidden']} | Dropout: {cfg_dict['dropout']:.2f}")
        print(f"  Head LR: {cfg_dict['head_lr']:.2e} | BB Mult: {cfg_dict['backbone_lr_mult']:.2f}")
        
        scores = []
        for s in seeds:
            t0 = time.time()
            score = evaluate_candidate(vec, seed=s, proxy_epochs=5, proxy_fraction=0.35)
            elapsed = time.time() - t0
            scores.append(score)
            print(f"    Seed {s} -> Proxy Val Macro-F1: {score:.4f} ({elapsed:.1f}s)")
            
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        results[cfg_idx] = (mean_score, std_score)
        print(f"  Summary Config #{cfg_idx + 1}: Mean = {mean_score:.4f} ± {std_score:.4f}")

    overall_std = np.mean([res[1] for res in results.values()])
    print("\n==================================================")
    print(f"NOISE FLOOR BENCHMARK: Mean Std Dev = ±{overall_std:.4f}")
    if overall_std <= 0.02:
        print("RESULT: Noise floor is within acceptable limits (<= 0.02). Ready for Phase E.")
    else:
        print("WARNING: High variance detected. Increase proxy budget before proceeding.")
    print("==================================================")

if __name__ == "__main__":
    main()