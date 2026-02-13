import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
import json
from .config import PREDICTABILITY_MODEL_PATH, QUANTILE_MODEL_DIR, DATA_PATH, TWO_HIGH_COVERAGES
from .critic_model_def import PredictabilityLSTM, scale_seq_and_current, FORMATION_TO_IDX, FORMATION_LIST, SEQ_LEN, n_off_teams

class CriticSystem:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Predictability Model
        self.pred_model = PredictabilityLSTM()
        try:
            checkpoint = torch.load(PREDICTABILITY_MODEL_PATH, map_location=self.device)
            # Handle if saved as a dict with metadata or just state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.pred_model.load_state_dict(checkpoint['model_state_dict'])
                
                # Load Metadata if available (Critical for alignment with friend's code)
                if 'off_team_to_idx' in checkpoint:
                     self.off_team_to_idx = checkpoint['off_team_to_idx']
                     print(f"Loaded off_team_to_idx from checkpoint. Keys: {len(self.off_team_to_idx)}")
                if 'formation_to_idx' in checkpoint:
                     # Use this instead of global FORMATION_TO_IDX if present
                     self.formation_to_idx = checkpoint['formation_to_idx']
                     print(f"Loaded formation_to_idx from checkpoint.")
            else:
                self.pred_model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Warning: Could not load predictability model weights or metadata: {e}")
        self.pred_model.to(self.device)
        self.pred_model.eval()

        # Load Quantile Models (Expected Yards)
        self.quantile_models = {}
        try:
            for q in [10, 50, 90]:
                path = os.path.join(QUANTILE_MODEL_DIR, f"lgbm_quantile_q{q}.pkl")
                if os.path.exists(path):
                    self.quantile_models[f"q{q}"] = joblib.load(path)
            print(f"Loaded Quantile Models: {list(self.quantile_models.keys())}")
        except Exception as e:
            print(f"Warning: Failed to load quantile models: {e}")

        # Data for history/context
        # We still load the CSV for vocab fallback, but prioritize checkpoint metadata
        self.df = pd.read_csv(DATA_PATH, low_memory=False)
        if not hasattr(self, 'off_team_to_idx'):
            self.off_team_vocab = sorted(self.df['OffTeam'].dropna().unique().tolist())
            self.off_team_to_idx = {t: i for i, t in enumerate(self.off_team_vocab)}
        
        # Pre-compute DefAvg stats for fallback/filling (from notebook logic)
        self.global_avg_run = 4.1  # Approx from notebook
        self.global_avg_pass = 6.7 # Approx from notebook

    def evaluate(self, play_json, game_context, history_seq=None):
        print(f"\nProposed Play from LLM:\n{json.dumps(play_json, indent=2)}")
        
        # Helper to get key with fallbacks
        def get_val(keys, default=None):
            for k in keys:
                if k in play_json: return play_json[k]
            return default

        # Parse Formation from "Personnel & Formation" string
        raw_formation = get_val(["Personnel & Formation", "personnel_formation", "formation"], "Unknown")
        formation_name = self._parse_formation(raw_formation)
        
        # 1. Predictability Score
        routes = play_json.get("Route Responsibilities") or play_json.get("route_responsibilities") or {}
        pred_score, pred_details = self._get_predictability_score(formation_name, game_context, history_seq, routes)
        
        # 2. Expected Yards (Quantile Inference)
        exp_yards, exp_details, feature_vector = self._get_expected_yards(play_json, game_context, formation_name)
        
        # 3. Aggregated Critique
        critique_text = self._generate_feedback(pred_score, exp_yards, formation_name)
        
        return {
            "predictability_score": round(float(pred_score), 4),
            "expected_yards": float(exp_yards),
            "critique": critique_text,
            "details": {
                "pred_details": pred_details,
                "exp_details": exp_details,
                "features": feature_vector
            }
        }

    # Allowed ReceiverAlignments
    VALID_ALIGNMENTS = [
        'Spread', 'Slot Left', 'Slot Right', 'Trips Right', 'Trips Left',
        'Balanced', 'Bunch Right', 'Bunch Left', 'Goal Line'
    ]

    def _parse_formation(self, raw_string):
        """Extracts formation name and maps to strict allowed list."""
        if ',' in raw_string:
            # Assume implicit convention: "Personnel, Formation"
            candidate = raw_string.split(',')[-1].strip()
        else:
            candidate = raw_string.strip()
            
        candidate = candidate.replace("Gun ", "").replace("Pistol ", "").replace("Under Center ", "")
        
        # Fuzzy / Direct mapping
        c_lower = candidate.lower()
        if 'spread' in c_lower: return 'Spread'
        if 'trips right' in c_lower: return 'Trips Right'
        if 'trips left' in c_lower: return 'Trips Left'
        if 'bunch right' in c_lower: return 'Bunch Right'
        if 'bunch left' in c_lower: return 'Bunch Left'
        if 'slot right' in c_lower: return 'Slot Right'
        if 'slot left' in c_lower: return 'Slot Left'
        if 'goal' in c_lower: return 'Goal Line'
        if 'balanced' in c_lower: return 'Balanced'
        
        # Fallback based on text
        if 'trips' in c_lower: return 'Trips Right' 
        if 'bunch' in c_lower: return 'Bunch Right'
        
        return 'Spread' # Default fallback


    def _get_predictability_score(self, formation_name, game_context, history_seq, routes=None):
        # 1. Pre-Snap Score (Formation Probability)
        # Use checked metadata if available, else Global fallback
        if hasattr(self, 'formation_to_idx'):
             fmt_map_ref = self.formation_to_idx
        else:
             fmt_map_ref = FORMATION_TO_IDX
             
        fmt_map_norm = {k.lower().strip(): v for k, v in fmt_map_ref.items()}
        target_fmt_idx = fmt_map_norm.get(formation_name.lower().strip())
       
        if target_fmt_idx is None:
            # Fuzzy match
            for k, v in fmt_map_norm.items():
                if k in formation_name.lower() or formation_name.lower() in k:
                    target_fmt_idx = v
                    break
       
        down = int(game_context.get("down", 1))
        togo = int(game_context.get("distance", 10))
        field_pos = int(game_context.get("field_position_yards", 35))
        score_diff = int(game_context.get("score_diff", 0))
       
        gs = np.zeros(6, dtype=np.float32)
        gs[0] = (down - 1) / 3.0
        gs[1] = (togo - 1) / 18.0
        gs[2] = 0.0 # Sin Time
        gs[3] = 1.0 # Cos Time
        gs[4] = np.clip(score_diff, -40, 40) / 80.0
        gs[5] = field_pos / 100.0
       
        # --- CRITICAL FIX: SEQUENCE INITIALIZATION ---
        # We initialize with zeros for history, BUT we must set the OL feature (idx 3) to 0.5.
        # In training, OL is clipped(0,4,8)/8.0. For empty history (0), this becomes 4/8.0 = 0.5.
        # SEQ_LEN is assumed to be 8 based on training config
        seq = np.zeros((1, SEQ_LEN, 10), dtype=np.float32)
        seq[:, :, 3] = 0.5  
       
        dc = np.zeros((1, 25), dtype=np.float32)
       
        t_seq = torch.from_numpy(seq).to(self.device)
        t_dc = torch.from_numpy(dc).to(self.device)
        t_gs = torch.from_numpy(gs).unsqueeze(0).to(self.device)
        # Fallback to 0 if team not found (e.g. 'NE' vs 'NE ') -> Matches friend's behavior
        off_team_idx = self.off_team_to_idx.get(game_context.get("off_team_code", "NE"), 0)
        t_id = torch.tensor([off_team_idx], dtype=torch.long).to(self.device)
       
        with torch.no_grad():
            pass_logits, formation_logits = self.pred_model(t_seq, t_dc, t_gs, t_id)
            formation_probs = torch.softmax(formation_logits, dim=1).cpu().numpy()[0]
       
        if target_fmt_idx is not None:
            raw_score = formation_probs[target_fmt_idx]
            # Try to get name from formation list
            # We assume FORMATION_LIST matches indices. 
            match_name = FORMATION_LIST[target_fmt_idx] 
        else:
            raw_score = float(np.mean(formation_probs))
            match_name = "Unknown (Avg)"

        # Standardize to 0 based on global mean (0.127)
        # Negative = Unpredictable (Below Expected Freq)
        # Positive = Predictable (Above Expected Freq)
        standardized_score = raw_score - 0.127
       
        return float(standardized_score), {"match_name": match_name, "all_probs": formation_probs.tolist()}
       
    # Team Defense Stats (2024/2025 proj)
    TEAM_DEF_STATS = {
        'ARI': {'DefAvgRun': 4.215, 'DefAvgPass': 6.761}, 'ATL': {'DefAvgRun': 4.263, 'DefAvgPass': 6.127},
        'BAL': {'DefAvgRun': 3.931, 'DefAvgPass': 6.338}, 'BUF': {'DefAvgRun': 4.891, 'DefAvgPass': 5.518},
        'CAR': {'DefAvgRun': 4.180, 'DefAvgPass': 6.588}, 'CHI': {'DefAvgRun': 4.793, 'DefAvgPass': 7.005},
        'CIN': {'DefAvgRun': 4.989, 'DefAvgPass': 7.127}, 'CLE': {'DefAvgRun': 3.984, 'DefAvgPass': 4.945},
        'DAL': {'DefAvgRun': 4.411, 'DefAvgPass': 7.241}, 'DEN': {'DefAvgRun': 3.597, 'DefAvgPass': 4.942},
        'DET': {'DefAvgRun': 4.188, 'DefAvgPass': 6.284}, 'GB':  {'DefAvgRun': 4.034, 'DefAvgPass': 5.884},
        'HOU': {'DefAvgRun': 3.479, 'DefAvgPass': 5.726}, 'IND': {'DefAvgRun': 3.694, 'DefAvgPass': 6.414},
        'JAX': {'DefAvgRun': 3.572, 'DefAvgPass': 5.561}, 'KC':  {'DefAvgRun': 3.676, 'DefAvgPass': 6.370},
        'LAC': {'DefAvgRun': 3.960, 'DefAvgPass': 6.019}, 'LAR': {'DefAvgRun': 4.137, 'DefAvgPass': 5.832},
        'LV':  {'DefAvgRun': 3.792, 'DefAvgPass': 6.175}, 'MIA': {'DefAvgRun': 4.423, 'DefAvgPass': 6.776},
        'MIN': {'DefAvgRun': 3.771, 'DefAvgPass': 5.306}, 'NE':  {'DefAvgRun': 4.016, 'DefAvgPass': 5.915},
        'NO':  {'DefAvgRun': 3.531, 'DefAvgPass': 5.894}, 'NYG': {'DefAvgRun': 5.084, 'DefAvgPass': 6.013},
        'NYJ': {'DefAvgRun': 4.018, 'DefAvgPass': 6.941}, 'PHI': {'DefAvgRun': 3.978, 'DefAvgPass': 5.605},
        'PIT': {'DefAvgRun': 4.165, 'DefAvgPass': 6.122}, 'SEA': {'DefAvgRun': 3.389, 'DefAvgPass': 5.107},
        'SF':  {'DefAvgRun': 4.193, 'DefAvgPass': 6.410}, 'TB':  {'DefAvgRun': 4.101, 'DefAvgPass': 6.756},
        'TEN': {'DefAvgRun': 3.966, 'DefAvgPass': 7.055}, 'WAS': {'DefAvgRun': 4.471, 'DefAvgPass': 7.157},
    }

    def _get_expected_yards(self, play_json, game_context, formation_name="Unknown"):
        if not self.quantile_models:
            return 6.2, "Historical Average (Models missing)"

        # --- Helper Functions (User Provided Logic) ---
        def get_drop_depth(drop_type):
            if '3 step' in drop_type.lower(): return 3
            if '5 step' in drop_type.lower(): return 5
            if '7 step' in drop_type.lower(): return 7
            return 3
            
        def get_play_concept(drop_type):
            dt = drop_type.lower()
            if 'rpo' in dt: return 'RPO'
            if 'action' in dt: return 'Play Action'
            if 'screen' in dt: return 'Screen'
            return 'Standard' # Default

        def clean_coverage_type(cov, safeties):
            c_lower = cov.lower()
            if '0' in c_lower: return 'Cover 0'
            if '1' in c_lower: return 'Cover 1'
            if '2' in c_lower: return 'Cover 2'
            if '3' in c_lower: return 'Cover 3'
            if '4' in c_lower: return 'Cover 4'
            if '6' in c_lower: return 'Cover 6'
            return 'Cover 2' # Fallback

        def simplify_route(r):
            if not r: return 'No Route'
            r = r.lower()
            # Allowed: Cross, No Route, Vertical, Hook, Shallow, Out, Comeback, Post, Corner, Flat, Screen
            
            if 'no route' in r or 'block' in r: return 'No Route'
            if 'screen' in r: return 'Screen'
            if 'flat' in r or 'swing' in r or 'bubble' in r: return 'Flat'
            if 'corner' in r: return 'Corner'
            if 'post' in r: return 'Post'
            if 'comeback' in r: return 'Comeback'
            if 'out' in r: return 'Out'
            if 'hook' in r or 'curl' in r or 'hitch' in r: return 'Hook'
            if 'seam' in r or 'streak' in r or 'go' in r or 'fly' in r or 'vertical' in r or 'wheel' in r: return 'Vertical'
            if 'shallow' in r: return 'Shallow'
            if 'dig' in r or 'in' in r or 'over' in r or 'cross' in r: return 'Cross'
            if 'drag' in r: return 'Shallow' # Default Drag to Shallow
            if 'slant' in r: return 'Shallow' # Map Slant to Shallow
            
            return 'No Route' # Default fallback

        # --- Feature Construction ---
        routes = play_json.get("Route Responsibilities") or play_json.get("route_responsibilities") or {}
        
        # KEY MAPPING FIX: Ensure RB/TE are mapped to L/R slots if not explicitly provided
        # We fill from R2, L2, R3, L3 (Inner slots) outwards/inwards
        target_slots = ['R2', 'L2', 'R3', 'L3', 'R4', 'L4']
        legacy_keys = [k for k in routes.keys() if k not in ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4']]
        
        for k in legacy_keys:
            # Find first empty slot
            assigned = False
            for slot in target_slots:
                # Check if slot is missing OR is effectively empty/No Route
                val = str(routes.get(slot, "")).lower()
                is_free = (slot not in routes) or (not routes[slot]) or ('no route' in val) or ('block' in val)
                
                if is_free:
                    routes[slot] = routes[k]
                    assigned = True
                    break
            if not assigned:
                # Fallback to outer slots if inner are full (unlikely for 5 players)
                for slot in ['R1', 'L1']:
                    val = str(routes.get(slot, "")).lower()
                    is_free = (slot not in routes) or (not routes[slot]) or ('no route' in val) or ('block' in val)
                    
                    if is_free:
                        routes[slot] = routes[k]
                        break
        

        
        # 1. DistanceToGo (Logic: if Own side, 100 - start)
        start_yard = int(game_context.get("field_position_yards", 35))
        field_side = 'Own' if start_yard <= 50 else 'Opp'
        distance_to_go = (100 - start_yard) if field_side == 'Own' else start_yard
        
        # 2. Personnel (RB + TE strings)
        # Prioritize explicit key
        personnel_val = str(play_json.get("personnel", "")).strip()
        if len(personnel_val) == 2 and personnel_val.isdigit():
             personnel_feature = personnel_val
        else:
            # Fallback to parsing string
            personnel_str = play_json.get("Personnel & Formation", "11 Personnel").split(' ')[0]
            if len(personnel_str) == 2 and personnel_str.isdigit():
                personnel_feature = personnel_str
            else:
                personnel_feature = "11" 

        # 3. 2MinWarning
        qtr = int(game_context.get("quarter", 1))
        time_left = int(game_context.get("time_left", 900))
        is_2min = 1 if (qtr == 2 and time_left <= 120) else 0

        # 4. BoxCount (DL + LB)
        # Allow override from game_context, else default to 7
        box_count = int(game_context.get("box_count", 7))
        
        # 5. DropBack/Depth/Concept
        drop_type = "3 Step" 
        # Allow override for drop depth
        drop_depth = int(game_context.get("drop_depth", get_drop_depth(drop_type)))
        
        play_concept = get_play_concept(drop_type)
        dropback = 1 # Pass

        # 6. Coverage
        safeties = int(game_context.get("safeties", 2))
        raw_cov = game_context.get('coverage', 'Cover 2')
        cleaned_cov = clean_coverage_type(raw_cov, safeties)

        # 7. Defensive Stats
        def_team = game_context.get('def_team_code', 'MIN') # Default to MIN as per user scenario
        def_stats = self.TEAM_DEF_STATS.get(def_team, self.TEAM_DEF_STATS['MIN'])

        row = {
            'Down': int(game_context.get('down', 1)),
            'ToGo': int(game_context.get('distance', 10)),
            'DistanceToGo': distance_to_go,
            'TimeLeftQTR': time_left,
            'OffLeadBefore': 0,
            'QTR': qtr,
            '2MinWarning': is_2min,
            'Safeties': safeties,
            'BoxCount': box_count,
            'CoverageType': cleaned_cov,
            'Dropback':1,
            'DropDepth': drop_depth,
            'PlayConcept': play_concept,
            'Personnel': personnel_feature,
            'ReceiverAlignment': formation_name,
            # Routes
            'L1': simplify_route(routes.get('L1')),
            'L2': simplify_route(routes.get('L2')),
            'L3': simplify_route(routes.get('L3')),
            'L4': simplify_route(routes.get('L4')),
            'R4': simplify_route(routes.get('R4')),
            'R3': simplify_route(routes.get('R3')),
            'R2': simplify_route(routes.get('R2')),
            'R1': simplify_route(routes.get('R1')),
            # DefAvg stats
            'DefAvgRun': def_stats['DefAvgRun'],
            'DefAvgPass': def_stats['DefAvgPass']
        }
        
        df_input = pd.DataFrame([row])
        
        # Inference
        try:
            # We use q50 as the "Expected" value
            pred_median = self.quantile_models['q50'].predict(df_input)[0]
            
            # Optional: Get range
            pred_low = self.quantile_models.get('q10', self.quantile_models['q50']).predict(df_input)[0]
            pred_high = self.quantile_models.get('q90', self.quantile_models['q50']).predict(df_input)[0]
            
            return round(pred_median, 2), f"Range: {pred_low:.1f} - {pred_high:.1f}", row
        except Exception as e:
            print(f"Inference Error: {e}")
            return 5.0, f"Error: {e}", {}

    def _generate_feedback(self, pred_score, exp_yards, formation):
        feed = []
        if pred_score > 0.0: # Lowered threshold for "high" (Normalized 0 = Avg)
            feed.append(f"STALE: Predictability ({pred_score:.2f}) is above average. Defense is ready for {formation}.")
        if exp_yards < 4.4:
            feed.append(f"INEFFICIENT: Expected Yards ({exp_yards:.1f}) is too low.")
        
        if not feed:
            return "Good. High efficiency and low predictability."
        return " ".join(feed)
