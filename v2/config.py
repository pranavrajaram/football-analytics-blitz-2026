import os

# Paths (relative to the v2 folder, so using .. to access root)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "2026_FAB_play_by_play.csv")
PREDICTABILITY_MODEL_PATH = os.path.join(ROOT_DIR, "predictability_lstm.pt")
EXPECTED_YARDS_MODEL_PATH = os.path.join(ROOT_DIR, "expected_yards.pkl")
SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "system_prompt.txt")

# OpenAI
OPENAI_MODEL = "gpt-4o"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    # Option to hardcode if env var not set (e.g. for testing)
    OPENAI_API_KEY = ""
    print("Using hardcoded API key.")

# Football Constants
TWO_HIGH_COVERAGES = {"cover 2", "cover 4", "man cover 2"}
QUANTILE_MODEL_DIR = os.path.join(ROOT_DIR, "quantile_model")
