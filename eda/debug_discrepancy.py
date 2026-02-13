import torch
import numpy as np
import pandas as pd
from v2.critic import CriticSystem

def brute_force_search():
    critic = CriticSystem()
    target_score = -0.067
    target_raw = target_score + 0.127 # 0.060
    
    print(f"Target Predictability: {target_score}")
    
    formation = "Trips Right"
    overrides = [0, 21] 
    
    # Defaults
    base_context = {
        "down": 1,
        "distance": 10,
        "field_position_yards": 35,
        "off_team_code": "NE"
    }

    original_map = critic.off_team_to_idx
    
    for idx_val in overrides:
        print(f"\nTesting Forced Index: {idx_val}")
        # forcing 'NE' to map to this index
        critic.off_team_to_idx = {'NE': idx_val}
        
        ctx = base_context.copy()
        score, details = critic._get_predictability_score(formation, ctx, None)
        print(f"Index {idx_val} Score: {score:.4f}")
        print(f"Index {idx_val} Prob: {details['all_probs'][4]:.4f}")
        
    critic.off_team_to_idx = original_map
    
if __name__ == "__main__":
    brute_force_search()
