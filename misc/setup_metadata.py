import pandas as pd
import joblib
import os

print("Generating metadata from training logic...")

if not os.path.exists('2026_FAB_play_by_play.csv'):
    print("❌ Error: CSV not found.")
else:
    df = pd.read_csv('2026_FAB_play_by_play.csv', low_memory=False)

    # 1. Rebuild Formations
    formation_counts = df['ReceiverAlignment'].fillna('Other').value_counts()
    main_formations = formation_counts[formation_counts >= 100].index.tolist()
    FORMATION_LIST = [f for f in main_formations if f not in ('Other', 'Kneel Down')]
    
    # 2. Rebuild Teams
    off_team_vocab = sorted(df['OffTeam'].dropna().unique().tolist())

    # 3. Create Metadata
    metadata = {
        'FORMATION_LIST': FORMATION_LIST,
        'formation_to_idx': {f: i for i, f in enumerate(FORMATION_LIST)},
        'off_team_vocab': off_team_vocab,
        'off_team_to_idx': {t: i for i, t in enumerate(off_team_vocab)},
        
        # EXACT DIMENSIONS FROM YOUR SCRIPT
        'SEQ_INPUT_DIM': 10,       # RB, WR, TE, OL, P_Type, P_Yds, P_EPA, Matchup, SymDelta, FamIdx
        'DRIVE_CONTEXT_STATIC': 11, # Rolling(5) + TimeSince + PrevCov + Holistic(4)
        'CURRENT_GAME_STATE_DIM': 6 
    }

    joblib.dump(metadata, 'predictability_metadata.pkl')
    print("✅ Metadata saved.")