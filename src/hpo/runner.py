import os
import json
import time
from src.hpo.objective import evaluate_candidate

class HPORunner:
    def __init__(self, log_file, seed):
        self.log_file = log_file
        self.seed = seed
        self.history = {} # Maps stringified vector to objective score
        self.best_score = 0.0
        self.best_vec = None
        self._load_history()

    def _load_history(self):
        if not os.path.exists(self.log_file):
            return
        with open(self.log_file, 'r') as f:
            for line in f:
                record = json.loads(line)
                vec_str = str(record['vector'])
                self.history[vec_str] = record['score']
                if record['score'] > self.best_score:
                    self.best_score = record['score']
                    self.best_vec = record['vector']
        print(f"Loaded {len(self.history)} previous evaluations from {self.log_file}")

    def evaluate(self, vector):
        vec_list = [float(v) for v in vector]
        vec_str = str(vec_list)
        
        # Return cached result if already evaluated (crash recovery)
        if vec_str in self.history:
            return self.history[vec_str]
            
        print(f"\n[EVAL] Testing vector: {[round(v, 3) for v in vec_list]}")
        
        t0 = time.time()
        # 5 epochs, 35% data - same as Phase D
        score = evaluate_candidate(vec_list, seed=self.seed, proxy_epochs=5, proxy_fraction=0.35) 
        elapsed = time.time() - t0
        
        record = {
            'timestamp': time.time(),
            'vector': vec_list,
            'score': score,
            'time_seconds': elapsed
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
            
        self.history[vec_str] = score
        if score > self.best_score:
            self.best_score = score
            self.best_vec = vec_list
            print(f"*** NEW BEST: {score:.4f} ***")
            
        return score