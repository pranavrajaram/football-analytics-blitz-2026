# Football Analytics Blitz 2026

UC-San Diego's submission to the Syracuse Football Analytics Blitz Data Competition, sponsored by SIS. 

Contributors: Jack Kalsched, Leo Malott, Diego Osborn, Pranav Rajaram, Adrian Rodriguez

---

## How to Access Our Web Visualization

Our **React + Vite** web app provides interactive visualizations for formation selection, route concepts, and two-high attack metrics.

**Prerequisites:** Node.js and npm (or yarn).

1. **Navigate to the web app directory:**
   ```bash
   cd football-analytics-blitz-2026/web_viz
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Run the development server:**
   ```bash
   npm run dev
   ```

4. **Open in your browser:**  
   Vite will print a local URL (e.g. `http://localhost:5173`). Open that URL to view the web visualization.

**Build for production:**
   ```bash
   npm run build
   npm run preview   # optional: preview the production build locally
   ```

---

## How to Access Our Dashboards

Our **Sean McVAI: Beating 2-High** Streamlit dashboard combines predictability scoring, expected yards (quantile models), and scenario exploration in one interface.

**Prerequisites:** Python with `streamlit` and project dependencies (see notebooks/requirements). Ensure quantile models and the predictability LSTM checkpoint are available in the paths expected by the app (e.g. `quantile_model/`, `predictability_lstm.pkl` when run from the app’s working directory).

1. **From the project root (or the directory that contains the model files):**
   ```bash
   cd football-analytics-blitz-2026
   streamlit run streamlit/demo_streamlit.py
   ```

2. **Open in your browser:**  
   Streamlit will open a browser tab (typically `http://localhost:8501`). Use the sidebar and controls to explore predictability, expected yards, and two-high attack strategies.

**Tip:** If the app reports missing files, run it from the directory that contains `quantile_model/` and `predictability_lstm.pkl`, or adjust paths inside `demo_streamlit.py` to match your layout.

---

## Methodology: LLM Feedback Loop Framework

We use a **closed-loop system** where an LLM proposes plays and a **critic** (data-driven models) scores them; the LLM then refines proposals using that feedback. This ties play design directly to predictability and expected yards.

### Architecture

1. **Generator (LLM)**  
   - **Location:** `v2/play_generator.py`  
   - **Role:** Given game context (down, distance, field position, defense personnel/coverage) and **memory of past attempts**, the LLM proposes a play (formation, personnel, route responsibilities) via a structured JSON API.  
   - **Memory:** Previous plays plus their critic scores and feedback are injected into the prompt so the LLM avoids repeating high-predictability or low–expected-yards designs.

2. **Critic (Models)**  
   - **Location:** `v2/critic.py`  
   - **Role:** For each proposed play, the critic returns:  
     - **Predictability score** (from the trained predictability LSTM): lower is more “unpredictable” and preferred.  
     - **Expected yards** (from quantile/expected-value models): higher is better.  
   - **Output:** A short textual **critique** plus these two scores, used as feedback for the next iteration.

3. **Self-Reflection (LLM)**  
   - **Location:** `play_generator.analyze_feedback()` in `v2/play_generator.py`  
   - **Role:** After each critique, the LLM is asked to explain *why* the critic gave those scores (e.g. formation tendency, route spacing) and to synthesize a “lesson learned” for the next iteration. This rationale is stored in memory and can be used to steer the next proposal.

4. **Orchestrator (Loop)**  
   - **Location:** `v2/orchestrator.py`  
   - **Flow:**  
     - **Generate** → **Critique** → **Rationale** → append to **memory** → repeat for a fixed number of iterations (e.g. 10).  
     - After the loop, the best play is selected by a composite score (e.g. Expected Yards − λ × Predictability).  
   - **Output:** A chosen play, full trace of iterations, and a summary; the trace is saved to `v2_feedback_trace.json` for inspection.

### How to Run the Feedback Loop

**Prerequisites:**  
- Python environment with project dependencies (PyTorch, OpenAI client, etc.).  
- Configured `OPENAI_API_KEY` (and any config in `v2/config.py`: model name, paths to predictability and quantile models).

1. **From the project root (so that `v2` and data paths resolve):**
   ```bash
   cd football-analytics-blitz-2026
   python -m v2.orchestrator
   ```
   Or, if you run from inside `v2`:
   ```bash
   cd football-analytics-blitz-2026/v2
   python orchestrator.py
   ```
   (Ensure `config.py` paths point to the correct model and data files.)

2. **Inspect results:**  
   Open `v2_feedback_trace.json` to see each iteration’s proposed play, critic scores, critique, and LLM rationale. The final selected play and composite score are printed to the console.

### Design Choices (Summary)

- **Predictability** is used so that “obvious” formations/tendencies are penalized and the offense stays hard to read.  
- **Expected yards** keeps plays grounded in estimated value.  
- **Memory + rationale** turns the loop into a **methodology**: the LLM explicitly reasons over critic feedback and avoids repeating the same mistakes, aligning play design with both analytics (critic) and strategic narrative (LLM).

---

## Repository Overview

| Component            | Path / entrypoint                    | Description                                      |
|---------------------|--------------------------------------|--------------------------------------------------|
| Web visualization   | `web_viz/` → `npm run dev`           | React + Vite front-end for visualizations       |
| Streamlit dashboard | `streamlit/demo_streamlit.py`        | “Sean McVAI: Beating 2-High” interactive app    |
| LLM feedback loop   | `v2/orchestrator.py`                 | Generator → Critic → Rationale → memory loop     |
| Play generator      | `v2/play_generator.py`              | LLM play proposals and feedback analysis       |
| Critic system       | `v2/critic.py`                       | Predictability + expected-yards scoring         |
| Notebooks           | `eda/`, `predictability.ipynb`, etc. | Analysis, model training, and exports           |

For data and model training details, see the notebooks and `eda/` (and any `requirements.txt` or environment files in the repo).
