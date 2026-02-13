"""
Generate a Patriots vs Vikings two-high attack play using the OpenAI API,
validate the JSON output, and compute expected-yards plus predictability
scores derived from the LSTM model outputs.

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
from typing import Dict, Tuple, List

from openai import OpenAI

DATA_PATH = "2026_FAB_play_by_play.csv"
SYSTEM_PROMPT_PATH = "patriots_vs_vikings_llm_system_prompt.txt"
INPUT_PATH = "patriots_vs_vikings_llm_input.json"
SCHEMA_PATH = "patriots_vs_vikings_llm_schema.json"

OUTPUT_JSON = "patriots_vs_vikings_llm_output.json"
REPORT_JSON = "patriots_vs_vikings_play_report.json"
PREDICTABILITY_SCORES_PATH = "predictability_scores.csv"

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


def defense_personnel_key(row: Dict[str, str]) -> str:
    dl = _to_int(row.get("DL"))
    lb = _to_int(row.get("LB"))
    db = _to_int(row.get("DB"))
    saf = _to_int(row.get("Safeties"))
    return f"DL{dl}-LB{lb}-DB{db}-S{saf}"


def extract_route_for_target(row: Dict[str, str]) -> str:
    tgt = (row.get("TargetedPlayer") or "").strip()
    if tgt.lower() == "back":
        return "BACK"
    if tgt in {"L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"}:
        return (row.get(tgt) or "").strip()
    return "UNKNOWN"


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
    expected_keys = {"L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"}
    if set(routes.keys()) != expected_keys:
        return False, "routes must include exactly: L1, L2, L3, L4, R4, R3, R2, R1"

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


def field_zone_from_start_yard(start_yard: int) -> str:
    if start_yard <= 20:
        return "backed_up"
    if start_yard <= 49:
        return "own_territory"
    if start_yard <= 60:
        return "midfield"
    if start_yard <= 80:
        return "plus_territory"
    return "red_zone"


def _normalize_text(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (x or "").strip().lower()).strip()


def _resolve_formation_label(llm_formation: str, formation_list: List[str]) -> Tuple[str, str]:
    """
    Match LLM formation text to ReceiverAlignment labels used by the model.
    Returns (matched_label, match_method).
    """
    if not llm_formation:
        return "UNKNOWN", "missing"
    norm = _normalize_text(llm_formation)
    if not norm:
        return "UNKNOWN", "missing"

    formation_map = {_normalize_text(f): f for f in formation_list if f}
    if norm in formation_map:
        return formation_map[norm], "exact_normalized"

    # Soft match using difflib
    import difflib
    candidates = difflib.get_close_matches(norm, formation_map.keys(), n=1, cutoff=0.6)
    if candidates:
        return formation_map[candidates[0]], "fuzzy"

    # Token overlap fallback
    norm_tokens = set(norm.split())
    best = ("UNKNOWN", 0)
    for k, original in formation_map.items():
        tokens = set(k.split())
        score = len(tokens & norm_tokens)
        if score > best[1]:
            best = (original, score)
    if best[1] > 0:
        return best[0], "token_overlap"

    return "UNKNOWN", "unmatched"


def _team_name_to_code(team_name: str) -> str:
    mapping = {
        "new england patriots": "NE",
        "minnesota vikings": "MIN",
        "kansas city chiefs": "KC",
        "buffalo bills": "BUF",
        "new england patriots": "NE",
    }
    key = (team_name or "").strip().lower()
    return mapping.get(key, (team_name or "").strip())


def _parse_field_position(field_position: str) -> int:
    """
    Extracts yardline (1-99) from strings like 'Own 35' or 'Opp 42'.
    Defaults to 0 on failure.
    """
    if not field_position:
        return 0
    m = re.search(r"(\d+)", field_position)
    if not m:
        return 0
    return _to_int(m.group(1), 0)


_PREDICTABILITY_CACHE = None


def _load_predictability_stats():
    """
    Builds aggregate stats from precomputed LSTM outputs in predictability_scores.csv.
    Returns dicts for contextual means and formation-level means.
    """
    global _PREDICTABILITY_CACHE
    if _PREDICTABILITY_CACHE is not None:
        return _PREDICTABILITY_CACHE

    by_context_formation = Counter()
    by_context_formation_n = Counter()
    by_context = Counter()
    by_context_n = Counter()
    by_formation = Counter()
    by_formation_n = Counter()
    overall_sum = 0.0
    overall_n = 0
    formations = set()

    with open(PREDICTABILITY_SCORES_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            formation = (row.get("ReceiverAlignment") or row.get("Actual_Formation") or "").strip()
            if not formation:
                continue
            formations.add(formation)

            try:
                p_score = float(row.get("P_score"))
            except Exception:
                continue

            down = _to_int(row.get("Down"))
            togo = _to_int(row.get("ToGo"))
            start_yard = _to_int(row.get("StartYard"))
            off = _clean_team(row.get("OffTeam"))
            if down <= 0 or togo <= 0 or start_yard <= 0:
                continue

            context = (
                off,
                down,
                togo_bucket(togo),
                field_zone_from_start_yard(start_yard),
            )
            key = context + (formation,)

            by_context_formation[key] += p_score
            by_context_formation_n[key] += 1
            by_context[context] += p_score
            by_context_n[context] += 1
            by_formation[formation] += p_score
            by_formation_n[formation] += 1
            overall_sum += p_score
            overall_n += 1

    _PREDICTABILITY_CACHE = {
        "by_context_formation": by_context_formation,
        "by_context_formation_n": by_context_formation_n,
        "by_context": by_context,
        "by_context_n": by_context_n,
        "by_formation": by_formation,
        "by_formation_n": by_formation_n,
        "overall_sum": overall_sum,
        "overall_n": overall_n,
        "formations": sorted(formations),
    }
    return _PREDICTABILITY_CACHE


def predictability_model_score(play: Dict, payload: Dict) -> Dict:
    """
    Uses LSTM model outputs (precomputed in predictability_scores.csv) to estimate
    the predictability of the LLM-play formation in the requested context.
    """
    stats = _load_predictability_stats()
    formations = stats["formations"]

    llm_formation = play.get("formation") or ""
    matched_formation, match_method = _resolve_formation_label(llm_formation, formations)

    down = _to_int(payload.get("down"), 1)
    togo = _to_int(payload.get("distance"), 10)
    start_yard = _parse_field_position(payload.get("field_position", "")) or 35
    off_team = _team_name_to_code(payload.get("offense_team", "NE")) or "NE"

    context = (
        off_team,
        down,
        togo_bucket(togo),
        field_zone_from_start_yard(start_yard),
    )

    # Primary: full context + formation
    key = context + (matched_formation,)
    sum_cf = stats["by_context_formation"].get(key, 0.0)
    n_cf = stats["by_context_formation_n"].get(key, 0)
    if n_cf > 0:
        p_score = sum_cf / n_cf
        ctx_sum = stats["by_context"].get(context, 0.0)
        ctx_n = stats["by_context_n"].get(context, 0)
        ctx_mean = ctx_sum / ctx_n if ctx_n > 0 else (stats["overall_sum"] / max(stats["overall_n"], 1))
        return {
            "method": "lstm_precomputed",
            "formation_llm": llm_formation,
            "formation_matched": matched_formation,
            "formation_match_method": match_method,
            "context_key": {
                "off_team": off_team,
                "down": down,
                "togo_bucket": togo_bucket(togo),
                "field_zone": field_zone_from_start_yard(start_yard),
            },
            "p_score": round(p_score, 3),
            "relative_predictability": round(p_score - ctx_mean, 3),
            "support": n_cf,
            "match_level": "context+formation",
        }

    # Fallback 1: formation only
    sum_f = stats["by_formation"].get(matched_formation, 0.0)
    n_f = stats["by_formation_n"].get(matched_formation, 0)
    if n_f > 0:
        p_score = sum_f / n_f
        overall_mean = stats["overall_sum"] / max(stats["overall_n"], 1)
        return {
            "method": "lstm_precomputed",
            "formation_llm": llm_formation,
            "formation_matched": matched_formation,
            "formation_match_method": match_method,
            "context_key": {
                "off_team": off_team,
                "down": down,
                "togo_bucket": togo_bucket(togo),
                "field_zone": field_zone_from_start_yard(start_yard),
            },
            "p_score": round(p_score, 3),
            "relative_predictability": round(p_score - overall_mean, 3),
            "support": n_f,
            "match_level": "formation_only",
        }

    # Final fallback: overall mean
    overall_mean = stats["overall_sum"] / max(stats["overall_n"], 1)
    return {
        "method": "lstm_precomputed",
        "formation_llm": llm_formation,
        "formation_matched": matched_formation,
        "formation_match_method": match_method,
        "context_key": {
            "off_team": off_team,
            "down": down,
            "togo_bucket": togo_bucket(togo),
            "field_zone": field_zone_from_start_yard(start_yard),
        },
        "p_score": round(overall_mean, 3),
        "relative_predictability": 0.0,
        "support": stats["overall_n"],
        "match_level": "overall_mean",
    }


def find_similar_plays(play: Dict, payload: Dict, top_k: int = 5) -> List[Dict]:
    """
    Finds top-K similar plays (any team vs two-high) using simple feature overlap:
    down, togo_bucket, field_zone, defensive personnel, formation match, targeted route match.
    """
    down = _to_int(payload.get("down"), 1)
    togo = _to_int(payload.get("distance"), 10)
    start_yard = _parse_field_position(payload.get("field_position", "")) or 35
    def_personnel = (payload.get("defense_personnel") or "").strip()

    y100 = start_yard
    context_togo = togo_bucket(togo)
    context_zone = field_zone(y100)

    llm_primary_route = infer_primary_route_label(play)
    llm_formation = play.get("formation") or ""

    # Build formation list from data for matching
    formation_set = set()
    with open(DATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ra = (row.get("ReceiverAlignment") or "").strip()
            if ra:
                formation_set.add(ra)
    formation_list = sorted(formation_set)
    matched_formation, _ = _resolve_formation_label(llm_formation, formation_list)

    scored = []
    with open(DATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not is_two_high(row.get("CoverageType")):
                continue

            r_down = _to_int(row.get("Down"))
            r_togo = _to_int(row.get("ToGo"))
            r_y100 = yardline_100(row.get("FieldSide"), row.get("StartYard"))
            if r_down <= 0 or r_togo <= 0 or r_y100 <= 0:
                continue

            score = 0.0
            if r_down == down:
                score += 2.0
            if togo_bucket(r_togo) == context_togo:
                score += 1.5
            if field_zone(r_y100) == context_zone:
                score += 1.0

            if def_personnel:
                if defense_personnel_key(row) == def_personnel:
                    score += 1.5

            ra = (row.get("ReceiverAlignment") or "").strip()
            if matched_formation and ra == matched_formation:
                score += 1.5

            if _to_int(row.get("Attempt")) == 1:
                route = extract_route_for_target(row)
                if route and route == llm_primary_route:
                    score += 1.0

            if score <= 0:
                continue

            scored.append({
                "score": round(score, 3),
                "Season": row.get("Season"),
                "Wk": row.get("Wk"),
                "OffTeam": row.get("OffTeam"),
                "DefTeam": row.get("DefTeam"),
                "Down": r_down,
                "ToGo": r_togo,
                "FieldSide": row.get("FieldSide"),
                "StartYard": row.get("StartYard"),
                "CoverageType": row.get("CoverageType"),
                "ReceiverAlignment": ra,
                "TargetedRoute": extract_route_for_target(row),
                "YardsOnPlay": row.get("YardsOnPlay"),
                "EPA": row.get("EPA"),
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def main():
    system_prompt = open(SYSTEM_PROMPT_PATH).read()
    payload = json.load(open(INPUT_PATH))
    schema = json.load(open(SCHEMA_PATH))

    # Strongly request JSON-only output
    user_content = (
        "Return valid JSON only. Use the provided input payload to generate one play. "
        "Do not include any extra commentary. "
        "Output ONLY these fields: formation, routes, rationale. "
        "Include a 'rationale' field (1–2 sentences) that explains the safety bind and progression. "
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
    predictability = predictability_model_score(play, payload)
    penalty = round(1.0 - predictability["p_score"], 3)

    report = {
        "play": play,
        "expected_yards": expected,
        "predictability_model": predictability,
        "predictability_penalty": penalty,
        "final_score": round(expected["expected_yards"] - penalty, 3),
    }

    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {REPORT_JSON}")


if __name__ == "__main__":
    main()
