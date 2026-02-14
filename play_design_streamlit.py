import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
import os

# --- 1. CONFIGURATION & LOADING ---
st.set_page_config(page_title="PlayCaller AI: Beat the 2-High", layout="wide")

# Cache models
@st.cache_resource
def load_models():
    quantiles = [0.02, 0.1, 0.5, 0.9, 0.97]
    models = {}
    try:
        for q in quantiles:
            path = f'quantile_model/lgbm_quantile_q{int(q*100)}.pkl'
            if os.path.exists(path):
                models[f'q{int(q*100)}'] = joblib.load(path)
            else:
                st.warning(f"Model file not found: {path}. Predictions will be disabled.")
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

# Load defensive stats
@st.cache_data
def load_team_stats():
    # Full 32 Team Data (2025-2026 Sample)
    data = {
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
    return data

models = load_models()
team_stats = load_team_stats()

# --- 2. SIDEBAR: FIXED GAME CONTEXT ---
st.sidebar.header("1. Game Situation")

def_team = st.sidebar.selectbox("Opponent", options=sorted(list(team_stats.keys())), index=20) # Default MIN
down = st.sidebar.selectbox("Down", [1, 2, 3, 4], index=0)
distance = st.sidebar.number_input("Yards to Go", min_value=1, max_value=99, value=10)
field_pos = st.sidebar.slider("Field Position", 1, 99, 35, help="1 = Own Goal Line, 99 = Opp Goal Line")

st.sidebar.markdown("---")
# --- FIXED DEFENSIVE LOOK ---
st.sidebar.markdown("### **DEFENSE: 2-HIGH SHELL**")
st.sidebar.caption("The defense is aligned in a split-safety structure. Adjust the specific variant and post-snap rotation below.")

# Only allow 2-High Variants
coverage_variant = st.sidebar.selectbox(
    "Coverage Variation", 
    options=['Cover 2', 'Cover 4', 'Man Cover 2'],
    index=0,
    help="'Cover 2' = Zone Flats. 'Cover 4' = Quarters. 'Man Cover 2' = 2-Man Under."
)

# Rotation / Disguise Sliders
st.sidebar.markdown("**Post-Snap Disguise**")
box_count = st.sidebar.slider("Box Count", 3, 9, 6, help="Standard 2-High box is 6 or 7. Higher numbers imply rotation.")
safeties = st.sidebar.slider("Deep Safeties (Post-Snap)", 0, 2, 2, help="Set to 1 to simulate a safety rotating down despite the 2-High shell call.")

# Context Dictionary
context_dict = {
    'Down': down,
    'ToGo': distance,
    'DistanceToGo': 100 - field_pos,
    'TimeLeftQTR': 900,
    'OffLeadBefore': 0,
    'QTR': 1,
    '2MinWarning': 0,
    'Safeties': safeties, # User can override this to simulate rotation
    'BoxCount': box_count,
    'CoverageType': coverage_variant, # Restricted to 2-High options
    'DefAvgRun': team_stats[def_team]['DefAvgRun'],
    'DefAvgPass': team_stats[def_team]['DefAvgPass'],
}

# --- 3. MAIN AREA: PLAY DESIGNER ---
st.title("🏈 Beat the 2-High: Game Planner")

col_settings, col_formation, col_results = st.columns([1, 1.5, 2])

# --- COLUMN 1: Basic Play Settings ---
with col_settings:
    st.subheader("Play Call")
    
    personnel_opts = ['11', '12', '21', '22', '20', '32', '13', '10', '01']
    concept_opts = ['Standard', 'Rollout', 'RPO', 'Screen', 'Trick']
    alignment_opts = ['Slot Left', 'Slot Right', 'Trips Right', 'Balanced', 'Twin Right', 'Single Left', 'Trips Left', 'All Tight', 'Single Right', 'Bunch Right', 'Kneel Down', 'Twin Left', 'Spread', 'Goal Line', 'Bunch Left']
    
    personnel = st.selectbox("Personnel", personnel_opts)
    concept = st.selectbox("Concept", concept_opts)
    alignment = st.selectbox("Alignment", alignment_opts, index=12) # Default Spread
    drop_depth = st.select_slider("Drop Depth", options=[0, 1, 3, 5, 7], value=5)

# --- COLUMN 2: Spatial Route Assignment ---
with col_formation:
    st.subheader("Route Tree")
    st.caption("Assign routes (Left -> Right). 'No Route' = Blocking/Pass Pro.")

    # EXACT options from your training data
    route_opts = ['Cross', 'No Route', 'Vertical', 'Hook', 'Shallow', 'Out', 'Comeback', 'Post', 'Corner', 'Flat', 'Screen']
    
    # Spatial Inputs
    # Defaulting L1/R1 to 'Vertical' (Index 2) and others to 'No Route' (Index 1)
    l1_route = st.selectbox("L1 (Far Left)", route_opts, index=2) 
    l2_route = st.selectbox("L2 (Slot Left)", route_opts, index=1)
    l3_route = st.selectbox("L3 (Tight Left)", route_opts, index=1)
    l4_route = st.selectbox("L4 (Backfield L)", route_opts, index=1)
    
    st.markdown("--- center ---")
    
    r4_route = st.selectbox("R4 (Backfield R)", route_opts, index=1)
    r3_route = st.selectbox("R3 (Tight Right)", route_opts, index=1)
    r2_route = st.selectbox("R2 (Slot Right)", route_opts, index=1)
    r1_route = st.selectbox("R1 (Far Right)", route_opts, index=2) 

    # Route Counter
    active_routes = 0
    for r in [l1_route, l2_route, l3_route, l4_route, r4_route, r3_route, r2_route, r1_route]:
        if r != 'No Route':
            active_routes += 1
    
    if active_routes > 5:
        st.error(f"⚠️ Illegal Formation: {active_routes} Routes (Max 5)")
    else:
        st.success(f"Eligible Receivers: {active_routes}")

    sim_btn = st.button("Simulate vs 2-High", type="primary")

# --- COLUMN 3: Results ---
with col_results:
    if sim_btn and models:
        # No need for "clean()" function anymore since inputs match training data
        
        play_dict = context_dict.copy()
        play_dict.update({
            'Dropback': 1,
            'DropDepth': drop_depth,
            'PlayConcept': concept,
            'Personnel': personnel,
            'ReceiverAlignment': alignment,
            'L1': l1_route, 'L2': l2_route, 'L3': l3_route, 'L4': l4_route,
            'R4': r4_route, 'R3': r3_route, 'R2': r2_route, 'R1': r1_route,
        })
        
        input_df = pd.DataFrame([play_dict])
        
        # Predict
        preds = {}
        for name, model in models.items():
            preds[name] = model.predict(input_df)[0]
        
        median = preds['q50']
        ceiling = preds['q90']
        floor = preds['q10']
        
        st.markdown(f"### Expected Yards vs **{def_team}** ({coverage_variant})")
        
        # --- DYNAMIC METRIC LOGIC ---
        
        # Floor: Detect Sack vs Incomplete vs Safe Gain
        if floor < -1.0:
            floor_label = "Sack Risk ⚠️"
            floor_delta = f"{floor:.1f} yds"
            floor_color = "inverse" # Red
        elif floor <= 0.5: 
            floor_label = "Incomplete Risk"
            floor_delta = "0.0 yds"
            floor_color = "off" # Grey
        else:
            floor_label = "High % Comp"
            floor_delta = f"{floor:.1f} yds"
            floor_color = "normal" # Green

        # Ceiling: Detect Explosiveness
        if ceiling >= 20:
            ceil_label = "Deep Shot 🚀"
        elif ceiling >= 10:
            ceil_label = "Chunk Play"
        else:
            ceil_label = "Conservative"

        # Median: Compare to Sticks
        yards_vs_sticks = median - distance
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Floor (10th %)", floor_delta, floor_label, delta_color=floor_color)
        m2.metric("Median Exp.", f"{median:.1f}", f"{yards_vs_sticks:+.1f} vs Marker")
        m3.metric("Ceiling (90th %)", f"{ceiling:.1f}", ceil_label)

        # --- DISTRIBUTION PLOT ---
        x_quantiles = [0.02, 0.1, 0.5, 0.9, 0.97]
        y_yards = [preds['q2'], preds['q10'], preds['q50'], preds['q90'], preds['q97']]
        
        try:
            # 1. Ensure Monotonicity
            y_yards_sorted = sorted(y_yards)
            
            # 2. Check for "degenerate" predictions
            if len(set(y_yards)) < 3:
                st.warning("Prediction spread too narrow/flat for curve generation.")
            else:
                # 3. Create Smooth Curve
                x_smooth = np.linspace(min(y_yards_sorted) - 5, max(y_yards_sorted) + 5, 200)
                spline = make_interp_spline(y_yards_sorted, x_quantiles, k=2)
                cdf_smooth = spline(x_smooth)
                
                # FIX: Clip gradient at 0 so density is never negative
                pdf_smooth = np.maximum(0, np.gradient(cdf_smooth, x_smooth))
                
                # 4. Plot
                fig, ax = plt.subplots(figsize=(8, 4))
                
                # Fill Area
                ax.fill_between(x_smooth, pdf_smooth, alpha=0.3, color='#4F8BF9')
                ax.plot(x_smooth, pdf_smooth, color='#4F8BF9', lw=2)
                
                # Markers
                ax.axvline(median, color='k', linestyle='--', alpha=0.8, label=f'Median: {median:.1f}')
                ax.axvline(distance, color='#E0245E', linestyle=':', lw=2, label=f'Target: {distance}')
                
                # Highlight "Sack Zone" (Negative Yards)
                if min(x_smooth) < 0:
                    ax.axvspan(min(x_smooth), 0, color='#E0245E', alpha=0.1, label='Loss Risk')

                # Visible Y-Axis
                ax.set_ylabel("Probability Density", fontsize=10)
                ax.tick_params(axis='y', labelsize=8)

                ax.set_xlabel("Yards Gained")
                ax.set_title(f"Expected Yards vs {coverage_variant}")
                ax.legend(loc='upper right', fontsize='small')
                sns.despine(left=False) # Showing spine so you can see Y-axis
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Plotting Error (Statistical Anomaly): {e}")