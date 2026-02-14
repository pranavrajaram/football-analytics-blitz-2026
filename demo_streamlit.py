import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
import os
import torch
import torch.nn as nn
import pickle

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Sean McVAI: Beating 2-High", layout="wide")

# --- 2. MODEL DEFINITIONS ---
class PredictabilityLSTM(nn.Module):
    def __init__(self, seq_input_size, drive_context_size, current_game_state_size,
                 n_off_teams, embed_dim=8, hidden_size=64, n_formations=5, dropout=0.3):
        super().__init__()
        self.off_team_embed = nn.Embedding(n_off_teams, embed_dim)
        self.layer_norm = nn.LayerNorm(seq_input_size)
        self.lstm = nn.LSTM(seq_input_size, hidden_size, batch_first=True, dropout=dropout)
        
        combined_size = hidden_size + drive_context_size + current_game_state_size + embed_dim
        self.fc = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pass_head = nn.Linear(64, 1)
        self.formation_head = nn.Linear(64, n_formations)
        
        # Uncertainty parameters from training checkpoint
        self.log_var_pass = nn.Parameter(torch.zeros(1))
        self.log_var_form = nn.Parameter(torch.zeros(1))

    def forward(self, seq, drive_context, current_game_state, off_team_idx):
        seq = self.layer_norm(seq)
        team_emb = self.off_team_embed(off_team_idx)
        _, (h_n, _) = self.lstm(seq)
        combined = torch.cat([h_n[-1], drive_context, current_game_state, team_emb], dim=1)
        x = self.fc(combined)
        return self.pass_head(x).squeeze(-1), self.formation_head(x)

# --- 3. RESOURCE LOADING ---

@st.cache_resource
def load_all_resources():
    resources = {'quantiles': {}, 'predictability': None, 'meta': None}
    
    # A. Load Quantile Models (Expected Yards)
    quantiles = [0.02, 0.1, 0.5, 0.9, 0.97]
    try:
        for q in quantiles:
            path = f'quantile_model/lgbm_quantile_q{int(q*100)}.pkl'
            if os.path.exists(path):
                resources['quantiles'][f'q{int(q*100)}'] = joblib.load(path)
    except Exception as e:
        st.error(f"Quantile Model Error: {e}")

    # B. Load Predictability Model (AI Scouting)
    pred_path = 'predictability/predictability_lstm.pkl'
    if os.path.exists(pred_path):
        try:
            checkpoint = None
            try:
                checkpoint = torch.load(pred_path, map_location='cpu', weights_only=False)
            except Exception:
                with open(pred_path, 'rb') as f:
                    checkpoint = pickle.load(f)
            
            if checkpoint and isinstance(checkpoint, dict) and 'FORMATION_LIST' in checkpoint:
                config = checkpoint['config']
                meta = {
                    'FORMATION_LIST': checkpoint['FORMATION_LIST'],
                    'formation_to_idx': checkpoint['formation_to_idx'],
                    'off_team_vocab': list(checkpoint['off_team_to_idx'].keys()),
                    'off_team_to_idx': checkpoint['off_team_to_idx'],
                    'SEQ_INPUT_DIM': config['SEQ_INPUT_DIM'],
                    'DRIVE_CONTEXT_DIM': config['DRIVE_CONTEXT_DIM'],
                    'CURRENT_GAME_STATE_DIM': config['CURRENT_GAME_STATE_DIM']
                }
                
                model = PredictabilityLSTM(
                    seq_input_size=meta['SEQ_INPUT_DIM'],
                    drive_context_size=meta['DRIVE_CONTEXT_DIM'],
                    current_game_state_size=meta['CURRENT_GAME_STATE_DIM'],
                    n_off_teams=config['n_off_teams'],
                    n_formations=config['n_formations']
                )
                
                model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                model.eval()
                
                resources['predictability'] = model
                resources['meta'] = meta
        except Exception as e:
            st.error(f"Predictability Model Load Failed: {e}")
            
    return resources

@st.cache_data
def load_team_stats():
    return {
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

# Initialization
res = load_all_resources()
q_models = res['quantiles']
pred_model = res['predictability']
pred_meta = res['meta']
team_stats = load_team_stats()

# --- 4. SIDEBAR ---
st.sidebar.title("War Room Settings")
with st.sidebar.expander("System Readiness", expanded=True):
    if q_models: st.success("✅ Expected Yards Loaded")
    if pred_model: st.success("✅ AI Scout Active")
    else: st.error("❌ Models Missing or Corrupt")

st.sidebar.header("1. Game Context")
def_team = st.sidebar.selectbox("Opponent", sorted(list(team_stats.keys())), index=20)
down = st.sidebar.selectbox("Down", [1, 2, 3, 4], index=0)
distance = st.sidebar.number_input("Yards to Go", min_value=1, max_value=99, value=10)
field_pos = st.sidebar.slider("Field Position", 1, 99, 35)

st.sidebar.markdown("---")
st.sidebar.markdown("### **DEFENSE: 2-HIGH SHELL**")
coverage_variant = st.sidebar.selectbox("Variant", ['Cover 2', 'Cover 4', 'Man Cover 2'], index=0)
box_count = st.sidebar.slider("Box Count", 3, 9, 7)
safeties = st.sidebar.slider("Deep Safeties (Post-Snap)", 0, 2, 2)

st.sidebar.markdown("---")
st.sidebar.header("3. Offensive Identity")
target_teams = ['NE ', 'BUF', 'KC ']

# Filter the model's vocabulary to only include those three
if pred_meta:
    # We use a list comprehension to ensure we only show teams the model actually knows
    off_team_opts = [t for t in target_teams if t in pred_meta['off_team_vocab']]
else:
    off_team_opts = target_teams

# If for some reason the list is empty (model didn't load), fallback to target_teams
if not off_team_opts:
    off_team_opts = target_teams

# Display the selectbox. index=0 will now default to 'KC' (or whichever is first in your target_teams)
off_team = st.sidebar.selectbox("Offensive Unit", off_team_opts, index=0)
score_diff = st.sidebar.number_input("Offense Lead/Trail", value=0)

context_dict = {
    'Down': down, 'ToGo': distance, 'DistanceToGo': 100 - field_pos,
    'TimeLeftQTR': 900, 'OffLeadBefore': score_diff, 'QTR': 1, '2MinWarning': 0,
    'Safeties': safeties, 'BoxCount': box_count, 'CoverageType': coverage_variant,
    'DefAvgRun': team_stats[def_team]['DefAvgRun'],
    'DefAvgPass': team_stats[def_team]['DefAvgPass'],
}

# --- 5. MAIN AREA ---
st.title("🏈 Sean McVAI: Play Design Lab")
col_settings, col_formation, col_results = st.columns([1, 1.5, 2])

with col_settings:
    st.subheader("Strategy")
    personnel = st.selectbox("Personnel", ['11', '21', '12', '13', '01', '22', '20', '10', '02', '03', '04', '00', '32', '31', '14'])
    concept = st.selectbox("Concept", ['Standard', 'Rollout', 'RPO', 'Screen', 'Trick'])
    alignment = st.selectbox("Alignment", ['Spread', 'Slot Left', 'Slot Right', 'Trips Right', 'Trips Left', 'Balanced', 'Bunch Right', 'Bunch Left', 'Goal Line'], index=0)
    drop_depth = st.select_slider("Drop Depth", options=[0, 1, 3, 5, 7], value=5)

# --- COLUMN 2: Route Assignment ---
with col_formation:
    st.subheader("Route Tree")
    st.caption("Assign routes (Left -> Right). 'No Route' = Blocking.")
    route_opts = ['Cross', 'No Route', 'Vertical', 'Hook', 'Shallow', 'Out', 'Comeback', 'Post', 'Corner', 'Flat', 'Screen']
    
    # Capture all route inputs with unique variable names
    l1_r = st.selectbox("L1", route_opts, index=2)
    l2_r = st.selectbox("L2", route_opts, index=1)
    l3_r = st.selectbox("L3", route_opts, index=1)
    l4_r = st.selectbox("L4", route_opts, index=1)
    
    st.markdown("--- center ---")
    
    r4_r = st.selectbox("R4", route_opts, index=1)
    r3_r = st.selectbox("R3", route_opts, index=1)
    r2_r = st.selectbox("R2", route_opts, index=1)
    r1_r = st.selectbox("R1", route_opts, index=2) 

    # --- ELIGIBLE RECEIVER COUNTER ---
    # List all current selections
    all_routes = [l1_r, l2_r, l3_r, l4_r, r4_r, r3_r, r2_r, r1_r]
    # Count how many are NOT 'No Route'
    eligible_count = sum(1 for route in all_routes if route != 'No Route')
    
    # Visual Feedback & Button State
    if eligible_count > 6:
        st.error(f"⚠️ Illegal Formation: {eligible_count} Routes (NFL Max is 6)")
        sim_disabled = True
    else:
        st.success(f"Eligible Receivers: {eligible_count}/6")
        sim_disabled = False

    # Simulate button is disabled if validation fails
    sim_btn = st.button(
        "Simulate Play Outcome", 
        type="primary", 
        disabled=sim_disabled,
        help="You cannot have more than 5 players running routes."
    )

# --- COLUMN 3: Results ---
with col_results:
    if sim_btn and not sim_disabled and q_models:
        play_dict = context_dict.copy()
        play_dict.update({
            'Dropback': 1 if drop_depth > 0 else 0, 'DropDepth': drop_depth,
            'PlayConcept': concept, 'Personnel': personnel, 'ReceiverAlignment': alignment,
            'L1': l1_r, 'L2': l2_r, 'L3': l3_r, 'L4': l4_r, 
            'R4': r4_r, 'R3': r3_r, 'R2': r2_r, 'R1': r1_r
        })
        input_df = pd.DataFrame([play_dict])
        preds = {name: model.predict(input_df)[0] for name, model in q_models.items()}
        median, ceiling, floor = preds['q50'], preds['q90'], preds['q10']
        
        st.markdown(f"### Expected Yards vs **{def_team}** ({coverage_variant})")
        
        # Metric Logic
        if floor < -1.0: floor_lbl, floor_clr = "Sack Risk ⚠️", "inverse"
        elif floor <= 0.5: floor_lbl, floor_clr = "Incomplete Risk", "off"
        else: floor_lbl, floor_clr = "High % Comp", "normal"
        
        ceil_lbl = "Deep Shot 🚀" if ceiling >= 20 else "Chunk Play" if ceiling >= 10 else "Conservative"
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Floor (10th %)", f"{floor:.1f} yds")
        m2.metric("Median (50th %)", f"{median:.1f} yds", f"{median - distance:+.1f} to Goal")
        m3.metric("Ceiling (90th %)", f"{ceiling:.1f} yds")

        # Expected Yards Plot
        try:
            x_quantiles = [0.02, 0.1, 0.5, 0.9, 0.97]
            y_yards = sorted([preds['q2'], preds['q10'], preds['q50'], preds['q90'], preds['q97']])
            if len(set(y_yards)) > 2:
                x_smooth = np.linspace(min(y_yards)-5, max(y_yards)+10, 300)
                spline = make_interp_spline(y_yards, x_quantiles, k=2)
                # Clip gradient at 0 so density is never negative
                pdf_smooth = np.maximum(0, np.gradient(spline(x_smooth), x_smooth))
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.fill_between(x_smooth, pdf_smooth, alpha=0.3, color='#4F8BF9')
                ax.plot(x_smooth, pdf_smooth, color='#4F8BF9', lw=2)
                ax.axvline(median, color='k', linestyle='--', alpha=0.8, label=f'Median: {median:.1f}')
                ax.axvline(distance, color='#E0245E', linestyle=':', lw=2, label=f'Target: {distance}')
                
                if min(x_smooth) < 0:
                    ax.axvspan(min(x_smooth), 0, color='#E0245E', alpha=0.1, label='Loss Risk')

                ax.set_ylabel("Probability Density", fontsize=9)
                ax.set_yticks([])
                ax.set_xlabel("Yards Gained")
                ax.legend(loc='upper right', fontsize='small')
                sns.despine(left=True)
                st.pyplot(fig)
        except Exception: st.info("Outcome variance too low for visualization.")

        # --- UPDATED AI SCOUT ANALYSIS ---
        if pred_model and pred_meta:
            st.markdown("---")
            st.subheader(f"AI Scout Analysis: {off_team}")
            
            # 1. Prepare Inputs
            # FIX A (Shared Error): Set OL feature to 0.5 to match training scaling for "empty history"
            seq_input = torch.zeros((1, 8, pred_meta['SEQ_INPUT_DIM']))
            seq_input[:, :, 3] = 0.5  
            
            drive_context = torch.zeros((1, pred_meta['DRIVE_CONTEXT_DIM']))
            
            # FIX B (The Discrepancy): Use field_pos / 100.0 directly (Match Partner)
            # Also ensure score_diff matches their hardcoded 0.0 if you want exact parity.
            gs_vec = np.array([
                (down - 1) / 3.0, 
                (distance - 1) / 18.0,
                0.0, # sin(time)
                1.0, # cos(time)
                np.clip(score_diff, -40, 40) / 80.0,
                field_pos / 100.0  # <--- CHANGED: Removed "100 - " to match partner/training
            ], dtype=np.float32)

            with torch.no_grad():
                pass_l, form_l = pred_model(
                    seq_input, 
                    drive_context, 
                    torch.tensor(gs_vec).unsqueeze(0), 
                    torch.tensor([pred_meta['off_team_to_idx'].get(off_team, 0)])
                )
                p_prob = torch.sigmoid(pass_l).item()
                f_probs = torch.softmax(form_l, dim=1).numpy()[0]
            
            # 2. Predictability Score Calculation
            idx_f = pred_meta['formation_to_idx'].get(alignment)
            
            # Fuzzy match
            if idx_f is None:
                formation_lower = alignment.lower()
                for name, idx in pred_meta['formation_to_idx'].items():
                    if name.lower() in formation_lower or formation_lower in name.lower():
                        idx_f = idx
                        break
            
            if idx_f is not None:
                raw_score = f_probs[idx_f]
                match_name = pred_meta['FORMATION_LIST'][idx_f]
            else:
                raw_score = np.mean(f_probs)
                match_name = "Avg (Unknown Fm)"

            pred_idx = raw_score - 0.127
            
            # --- DISPLAY ---
            c1, c2, c3 = st.columns([1, 1, 1.5])
            
            c1.metric("Likely Play", "Pass" if p_prob > 0.5 else "Run")
            c2.metric("Likely Look", pred_meta['FORMATION_LIST'][np.argmax(f_probs)])
            
            if pred_idx > 0.10: 
                score_color = "inverse" 
                label = "High Predictability"
            elif pred_idx < -0.05: 
                score_color = "normal" 
                label = "Low Predictability"
            else: 
                score_color = "off"
                label = "Standard"

            c3.metric(
                "Predictability Index", 
                f"{pred_idx:.3f}", 
                delta=label,
                delta_color=score_color
            )