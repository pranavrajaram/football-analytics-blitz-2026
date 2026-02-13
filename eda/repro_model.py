import pandas as pd
import torch
import numpy as np
from v2.critic import CriticSystem

def test_specific_play():
    critic = CriticSystem()
    
    # User inputs
    # Play: 11 personnel, standard concept, spread alignment, 5 drop depth
    # Routes: L1 Vertical, R2 Out, R1 Post, L4 Shallow
    # Context: 2-high shell, cover 2, box 7, safeties 2
    
    play_json = {
        "personnel_formation": "11 Personnel, Spread",
        "route_responsibilities": {
            "L1": "Vertical",
            "L2": "None",
            "L3": "None",
            "L4": "Shallow",
            "R4": "None",
            "R3": "None",
            "R2": "Out",
            "R1": "Post",
            "RB": "Block",   # Implicit?
            "TE": "Block"    # Implicit?
        }
    }
    
    game_context = {
        "down": 1,
        "distance": 10,
        "field_position_yards": 35, # Ball on 35
        "coverage": "Cover 2",
        "off_team_code": "NE"
    }

    print("\n--- Testing Specific Play Configuration ---")
    
    # We need to manually construct the input features slightly because critic.py 
    # has some hardcodings (like BoxCount=6) that we need to override to match user request (Box=7).
    # So we will modify the _get_expected_yards logic temporarily or subclass it here for testing.
    
    # Let's inspect what the critic generates first with default parsing
    # But specifically forcing BoxCount=7 and DropDepth=5 in the internal extraction 
    # requires modifying critic.py or mocking.
    
    # Ideally, we call critic._get_expected_yards but intercept the DataFrame creation.
    # For now, let's copy the extraction logic and modify parameters to match user request exact feature vector.
    
    # Reconstruct the exact row the user wants:
    params = {
        'Down': 1,
        'ToGo': 10,
        'DistanceToGo': 65, # Own 35 -> 65 yds to go
        'TimeLeftQTR': 900,
        'OffLeadBefore': 0,
        'QTR': 1,
        '2MinWarning': 0,
        'Safeties': 2,
        'BoxCount': 7,          # USER REQUEST
        'CoverageType': 'Cover 2',
        'Dropback': 1,
        'DropDepth': 5,         # USER REQUEST
        'PlayConcept': 'Standard',
        'Personnel': '11',
        'ReceiverAlignment': 'Spread',
        'L1': 'Vertical',
        'L2': 'No Route',    # Was Other
        'L3': 'No Route',    # Was Other
        'L4': 'Shallow',     # Was Drag
        'R4': 'No Route',
        'R3': 'No Route',
        'R2': 'Out',
        'R1': 'Post',
        'DefAvgRun': 3.771,  # MIN Stats
        'DefAvgPass': 5.306  # MIN Stats
    }
    
    df_input = pd.DataFrame([params])
    
    print("Feature Vector:")
    print(df_input.iloc[0])
    
    # Predict Expected Yards
    if 'q50' in critic.quantile_models:
        pred_median = critic.quantile_models['q50'].predict(df_input)[0]
        pred_low = critic.quantile_models['q10'].predict(df_input)[0]
        pred_high = critic.quantile_models['q90'].predict(df_input)[0]
        print(f"\nExpected Yards (Median): {pred_median:.4f}")
        print(f"Range: {pred_low:.4f} - {pred_high:.4f}")
    else:
        print("Quantile models not loaded.")

    # Predict Predictability
    # For predictability, we need the sequences. 
    # The user said "PI of -0.09". 
    # Formation "Spread" -> Formation Index?
    # Context: 2-high shell...
    
    ps_score, details = critic._get_predictability_score("Spread", game_context, None)
    print(f"\nPredictability Score: {ps_score:.4f}")
    print(f"Match Name: {details['match_name']}")
    
if __name__ == "__main__":
    test_specific_play()
