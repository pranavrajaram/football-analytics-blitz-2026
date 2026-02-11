"""
Genei rae a Patriots vs Vikings two-high attack play using the OpenAI API,
validate the JSON output, and compute simple expected-yards and predictability
scores from the SIS play-by-play dataset.

Requires:
- OPENAI_API_KEY env var set
- openai Python SDK installed (pip install openai)

Outputs:
- patriots_vs_vikings_llm_output.json
- patriots_vs_vikings_play_report.json
"""

import csv
import json
import os
import re
from collections import Counter
from statistics import mean
from typing import Dict, Tuple

from openai import OpenAI

DATA_PATH = "2026_FAB_play_by_play.csv"
SYSTEM_PROMPT_PATH = "patriots_vs_vikings_llm_system_prompt.txt"
INPUT_PATH = "patriots_vs_vikings_llm_input.json"
SCHEMA_PATH = "patriots_vs_vikings_llm_schema.json"

OUTPUT_JSON = "patriots_vs_vikings_llm_output.json"
REPORT_JSON = "patriots_vs_vikings_play_report.json"

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")

TWO_HIGH_COVERAGES = {"cover 2", "cover 4", "man cover 2"}


def _to_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _clean_team(x: str) -> str:
    return (x or "").strip()


def is_two_high(cov: str) -> bool:
    return (cov or "").strip().lower() in TWO_HIGH_COVERAGES


def yardline_100(field_side: str, start_yard: str) -> int:
    side = (field_side or "").strip().lower()
    y = _to_int(start_yard, 0)
    if side == "own":
        return y
    if side == "oppo":
        return 100 - y
    return 0


def togo_bucket(togo: int) -> str:
    if togo <= 3:
        return "short"
    if togo <= 6:
        return "medium"
    if togo <= 10:
        return "long"
    return "very_long"


def field_zone(y100: int) -> str:
    if y100 <= 20:
        return "backed_up"
    if y100 <= 49:
        return "own_territory"
    if y100 <= 60:
        return "midfield"
    if y100 <= 80:
        return "plus_territory"
    return "red_zone"


def extract_json(text: str) -> Dict:
    """Best-effort JSON extraction from model output."""
    if not text:
        raise ValueError("No output text from model.")

    # Try direct JSON first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find the first JSON object
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Failed to parse JSON from model output.")


def validate_schema(play: Dict, schema: Dict) -> Tuple[bool, str]:
    """Lightweight schema validation without external deps."""
    required = schema.get("required", [])
    for key in required:
        if key not in play:
            return False, f"Missing required field: {key}"

    routes = play.get("routes")
    if not isinstance(routes, dict):
        return False, "routes must be an object with keys L1, L2, L3, L4, R4, R3, R2, R1"

    return True, "ok"


def infer_primary_route_label(play: Dict) -> str:
    """
    Attempts to resolve a representative route label for predictability scoring.
    Since we removed primary_read/secondary_read, pick the first available route.
    """
    routes = play.get("routes", {})
    for key in ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]:
        val = routes.get(key)
        if val:
            return str(val).strip()
    return "UNKNOWN"


def expected_yards_contextual(play: Dict) -> Dict:
    """
    Computes a simple expected-yards estimate based on similar historical plays
    by the Patriots vs two-high. Falls back to broader averages if needed.
    """
    personnel = play.get("formation", "")  # not used; we rely on input payload
    payload = json.load(open(INPUT_PATH))

    # Parse personnel string like "11" -> RB=1, TE=1, WR=3
    offense_personnel = payload.get("offense_personnel", "11")
    rb = int(offense_personnel[0]) if len(offense_personnel) > 0 else 1
    te = int(offense_personnel[1]) if len(offense_personnel) > 1 else 1
    wr = 5 - rb - te

    context_bucket = {
        "down": 1,
        "togo_bucket": "long",
        "field_zone": "own_territory",
        "rb": rb,
        "te": te,
        "wr": wr,
    }

    yards_context = []
    yards_overall = []

    with open(DATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if _clean_team(row.get("OffTeam")) != "NE":
                continue
            if not is_two_high(row.get("CoverageType")):
                continue

            y100 = yardline_100(row.get("FieldSide"), row.get("StartYard"))
            down = _to_int(row.get("Down"))
            togo = _to_int(row.get("ToGo"))
            if down <= 0 or togo <= 0 or y100 <= 0:
                continue

            if _to_int(row.get("RB")) != rb:
                continue
            if _to_int(row.get("TE")) != te:
                continue
            if _to_int(row.get("WR")) != wr:
                continue

            yards = _to_float(row.get("YardsOnPlay"))
            yards_overall.append(yards)

            if down == 1 and togo_bucket(togo) == "long" and field_zone(y100) == "own_territory":
                yards_context.append(yards)

    if yards_context:
        return {"expected_yards": round(mean(yards_context), 2), "sample_n": len(yards_context), "level": "context"}
    if yards_overall:
        return {"expected_yards": round(mean(yards_overall), 2), "sample_n": len(yards_overall), "level": "two_high_overall"}

    return {"expected_yards": 0.0, "sample_n": 0, "level": "none"}


def predictability_penalty(play: Dict) -> Dict:
    """
    Simple tendency penalty using route frequency in similar context (league-wide).
    penalty = 1 - P(route | context)
    """
    payload = json.load(open(INPUT_PATH))

    offense_personnel = payload.get("offense_personnel", "11")
    rb = int(offense_personnel[0]) if len(offense_personnel) > 0 else 1
    te = int(offense_personnel[1]) if len(offense_personnel) > 1 else 1
    wr = 5 - rb - te

    target_route = infer_primary_route_label(play)

    counts = Counter()

    with open(DATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            down = _to_int(row.get("Down"))
            togo = _to_int(row.get("ToGo"))
            y100 = yardline_100(row.get("FieldSide"), row.get("StartYard"))
            if down <= 0 or togo <= 0 or y100 <= 0:
                continue

            if down != 1:
                continue
            if togo_bucket(togo) != "long":
                continue
            if field_zone(y100) != "own_territory":
                continue
            if _to_int(row.get("RB")) != rb:
                continue
            if _to_int(row.get("TE")) != te:
                continue
            if _to_int(row.get("WR")) != wr:
                continue

            if _to_int(row.get("Attempt")) == 1:
                tgt = (row.get("TargetedPlayer") or "").strip()
                if tgt in {"L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"}:
                    route = (row.get(tgt) or "").strip()
                elif tgt.lower() == "back":
                    route = "BACK"
                else:
                    route = "UNKNOWN"
                counts[route] += 1

    total = sum(counts.values())
    if total == 0:
        return {"penalty": 0.25, "route": target_route, "support": 0}

    p = counts.get(target_route, 0) / total
    return {"penalty": round(1.0 - p, 3), "route": target_route, "support": total}


def main():
    system_prompt = open(SYSTEM_PROMPT_PATH).read()
    payload = json.load(open(INPUT_PATH))
    schema = json.load(open(SCHEMA_PATH))

    # Strongly request JSON-only output
    user_content = (
        "Return valid JSON only. Use the provided input payload to generate one play. "
        "Do not include any extra commentary. "
        "Output ONLY these fields: formation, personnel, routes, rationale. "
        "Include a 'rationale' field (1–2 sentences). "
        "The routes field MUST be an object with keys L1, L2, L3, L4, R4, R3, R2, R1 (all strings).\n\n"
        + json.dumps(payload)
    )

    client = OpenAI()
    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    output_text = response.output_text
    play = extract_json(output_text)
    ok, msg = validate_schema(play, schema)
    if not ok:
        raise ValueError(f"Schema validation failed: {msg}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(play, f, indent=2)

    expected = expected_yards_contextual(play)
    penalty = predictability_penalty(play)

    report = {
        "play": play,
        "expected_yards": expected,
        "predictability_penalty": penalty,
        "final_score": round(expected["expected_yards"] - penalty["penalty"], 3),
    }

    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {REPORT_JSON}")


if __name__ == "__main__":
    main()
