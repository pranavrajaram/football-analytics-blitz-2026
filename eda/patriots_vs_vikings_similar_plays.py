"""
Find the 5 most similar two-high plays to an existing LLM-designed play,
without re-running the LLM prompt.

Inputs:
- patriots_vs_vikings_llm_output.json (LLM play)
- patriots_vs_vikings_llm_input.json (context: down/distance/field_position/defense_personnel)

Output:
- patriots_vs_vikings_similar_plays.json
"""

import csv
import json
import re
from typing import Dict, List, Tuple

DATA_PATH = "2026_FAB_play_by_play.csv"
INPUT_PATH = "patriots_vs_vikings_llm_input.json"
PLAY_PATH = "patriots_vs_vikings_llm_output.json"
OUTPUT_PATH = "patriots_vs_vikings_similar_plays.json"

TWO_HIGH_COVERAGES = {"cover 2", "cover 4", "man cover 2"}


def _to_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _clean_team(x: str) -> str:
    return (x or "").strip()


def _normalize_text(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (x or "").strip().lower()).strip()


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


def _parse_field_position(field_position: str) -> int:
    if not field_position:
        return 0
    m = re.search(r"(\d+)", field_position)
    if not m:
        return 0
    return _to_int(m.group(1), 0)


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


def infer_primary_route_label(play: Dict) -> str:
    routes = play.get("routes", {})
    for key in ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]:
        val = routes.get(key)
        if val:
            return str(val).strip()
    return "UNKNOWN"


def _resolve_formation_label(llm_formation: str, formation_list: List[str]) -> Tuple[str, str]:
    if not llm_formation:
        return "UNKNOWN", "missing"
    norm = _normalize_text(llm_formation)
    if not norm:
        return "UNKNOWN", "missing"

    formation_map = {_normalize_text(f): f for f in formation_list if f}
    if norm in formation_map:
        return formation_map[norm], "exact_normalized"

    import difflib
    candidates = difflib.get_close_matches(norm, formation_map.keys(), n=1, cutoff=0.6)
    if candidates:
        return formation_map[candidates[0]], "fuzzy"

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


def find_similar_plays(play: Dict, payload: Dict, top_k: int = 5) -> List[Dict]:
    down = _to_int(payload.get("down"), 1)
    togo = _to_int(payload.get("distance"), 10)
    start_yard = _parse_field_position(payload.get("field_position", "")) or 35
    def_personnel = (payload.get("defense_personnel") or "").strip()

    y100 = start_yard
    context_togo = togo_bucket(togo)
    context_zone = field_zone(y100)

    llm_primary_route = infer_primary_route_label(play)
    llm_formation = play.get("formation") or ""

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
    payload = json.load(open(INPUT_PATH))
    play = json.load(open(PLAY_PATH))

    similar = find_similar_plays(play, payload, top_k=10)
    out = {
        "play": play,
        "context": payload,
        "similar_plays": similar,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
