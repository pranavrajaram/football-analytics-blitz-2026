"""
LLM-based play design setup for Patriots (offense) vs Vikings (defense)
Focus: Part 3 of Football Analytics Blitz prompt.

This script:
- Extracts Vikings' most common two-high personnel grouping
- Summarizes Patriots pass success vs two-high (targeted route level)
- Builds a structured LLM input payload for 1st-and-10 at own 35
- Provides simple expected-yards and predictability scoring utilities

Dependencies: standard library only.
"""

import csv
import json
from collections import Counter, defaultdict
from statistics import mean
from typing import Dict, Tuple, List

DATA_PATH = "2026_FAB_play_by_play.csv"

# Team codes (strip spaces in raw data)
OFF_TEAM = "NE"   # Patriots
DEF_TEAM = "MIN"  # Vikings

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
    """
    Converts FieldSide + StartYard into a 1-99 field position scale.
    Own 35 -> 35, Opp 35 -> 65.
    """
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


def extract_route_for_target(row: Dict[str, str]) -> str:
    """
    Maps TargetedPlayer (L1..R4) to its route label.
    Returns "BACK" for backfield targets.
    """
    tgt = (row.get("TargetedPlayer") or "").strip()
    if tgt.lower() == "back":
        return "BACK"
    if tgt in {"L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"}:
        return (row.get(tgt) or "").strip()
    return "UNKNOWN"


def build_summaries():
    vikings_two_high_personnel = Counter()
    vikings_two_high_coverages = Counter()

    patriots_route_stats = defaultdict(list)  # route -> list of yards
    patriots_route_epa = defaultdict(list)    # route -> list of epa

    # Context-based predictability counts
    context_playtype_counts = defaultdict(Counter)
    context_route_counts = defaultdict(Counter)

    with open(DATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            off = _clean_team(row.get("OffTeam"))
            deff = _clean_team(row.get("DefTeam"))
            cov = row.get("CoverageType")
            two_high = is_two_high(cov)

            # Vikings two-high personnel summary
            if deff == DEF_TEAM and two_high:
                dl = _to_int(row.get("DL"))
                lb = _to_int(row.get("LB"))
                db = _to_int(row.get("DB"))
                saf = _to_int(row.get("Safeties"))
                personnel_key = f"DL{dl}-LB{lb}-DB{db}-S{saf}"
                vikings_two_high_personnel[personnel_key] += 1
                vikings_two_high_coverages[(cov or "").strip()] += 1

            # Patriots vs two-high: passing success by targeted route
            if off == OFF_TEAM and two_high:
                attempt = _to_int(row.get("Attempt"))
                if attempt == 1:
                    route = extract_route_for_target(row)
                    yds = _to_float(row.get("YardsOnPlay"))
                    epa = _to_float(row.get("EPA"))
                    patriots_route_stats[route].append(yds)
                    patriots_route_epa[route].append(epa)

            # Build predictability counts (all offenses)
            down = _to_int(row.get("Down"))
            togo = _to_int(row.get("ToGo"))
            y100 = yardline_100(row.get("FieldSide"), row.get("StartYard"))
            if down > 0 and togo > 0 and y100 > 0:
                rb = _to_int(row.get("RB"))
                te = _to_int(row.get("TE"))
                wr = _to_int(row.get("WR"))
                personnel = f"{rb}{te}{wr}"
                context = (
                    down,
                    togo_bucket(togo),
                    field_zone(y100),
                    personnel,
                )
                event_type = (row.get("EventType") or "").strip().lower()
                context_playtype_counts[context][event_type] += 1

                if _to_int(row.get("Attempt")) == 1:
                    route = extract_route_for_target(row)
                    context_route_counts[context][route] += 1

    # Summaries
    top_vikings_personnel = vikings_two_high_personnel.most_common(5)
    top_vikings_coverages = vikings_two_high_coverages.most_common(5)

    patriots_route_summary = []
    for route, ys in patriots_route_stats.items():
        if len(ys) >= 25:  # avoid tiny samples
            patriots_route_summary.append({
                "route": route,
                "n": len(ys),
                "avg_yards": round(mean(ys), 2),
                "avg_epa": round(mean(patriots_route_epa[route]), 3) if patriots_route_epa[route] else None,
            })
    patriots_route_summary.sort(key=lambda x: (x["avg_epa"], x["avg_yards"]))
    patriots_route_summary = patriots_route_summary[::-1][:15]

    return {
        "top_vikings_two_high_personnel": top_vikings_personnel,
        "top_vikings_two_high_coverages": top_vikings_coverages,
        "patriots_top_routes_vs_two_high": patriots_route_summary,
        "context_playtype_counts": context_playtype_counts,
        "context_route_counts": context_route_counts,
    }


def predictability_penalty(context_counts: Counter, play_label: str) -> float:
    """
    Simple tendency penalty: higher penalty for highly predictable plays.
    penalty = 1 - P(play_label | context)
    """
    total = sum(context_counts.values())
    if total == 0:
        return 0.25  # mild penalty when context is unseen
    return 1.0 - (context_counts.get(play_label, 0) / total)


def build_llm_payload(summary: Dict) -> Dict:
    """
    Constructs a structured LLM input payload for Part 3 situation.
    """
    top_personnel = summary["top_vikings_two_high_personnel"]
    top_personnel = top_personnel[0][0] if top_personnel else "UNKNOWN"

    payload = {
        "offense_team": "New England Patriots",
        "defense_team": "Minnesota Vikings",
        "down": 1,
        "distance": 10,
        "field_position": "Own 35",
        "offense_personnel": "11",  # typical starting point; adjust with data
        "defense_personnel": top_personnel,
        "coverage_family": "Two-High",
        "constraints": [
            "avoid heavy run tells",
            "keep 2-high defense on the field",
            "use realistic Patriots personnel",
        ],
        "top_routes_vs_two_high": summary["patriots_top_routes_vs_two_high"],
    }
    return payload


def write_artifacts():
    summary = build_summaries()

    # Write summary insights
    with open("patriots_vs_vikings_summary.json", "w") as f:
        json.dump({
            "top_vikings_two_high_personnel": summary["top_vikings_two_high_personnel"],
            "top_vikings_two_high_coverages": summary["top_vikings_two_high_coverages"],
            "patriots_top_routes_vs_two_high": summary["patriots_top_routes_vs_two_high"],
        }, f, indent=2)

    # Write LLM payload
    payload = build_llm_payload(summary)
    with open("patriots_vs_vikings_llm_input.json", "w") as f:
        json.dump(payload, f, indent=2)

    # Write system prompt + schema
    system_prompt = (
        "You are an NFL offensive analyst. Generate a pass play designed to defeat two-high safety coverage.\n"
        "Constraints:\n"
        "- Do not use heavy personnel formations that would cause the defense to audible out of two-high.\n"
        "- The play must be realistic for the Patriots’ personnel.\n"
        "- Routes must be coherent with standard passing concepts.\n"
        "- The play must be executable on 1st-and-10 from the offense’s own 35.\n"
        "Return valid JSON following the schema provided."
    )
    with open("patriots_vs_vikings_llm_system_prompt.txt", "w") as f:
        f.write(system_prompt)

    schema = {
        "type": "object",
        "required": ["formation", "motion", "protection", "routes", "primary_read", "secondary_read", "rationale"],
        "properties": {
            "formation": {"type": "string"},
            "motion": {"type": "string"},
            "protection": {"type": "string"},
            "routes": {
                "type": "object",
                "properties": {
                    "X": {"type": "string"},
                    "Z": {"type": "string"},
                    "Y": {"type": "string"},
                    "F": {"type": "string"},
                    "RB": {"type": "string"}
                }
            },
            "primary_read": {"type": "string"},
            "secondary_read": {"type": "string"},
            "rationale": {"type": "string"}
        }
    }
    with open("patriots_vs_vikings_llm_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    print("Wrote:")
    print("- patriots_vs_vikings_summary.json")
    print("- patriots_vs_vikings_llm_input.json")
    print("- patriots_vs_vikings_llm_system_prompt.txt")
    print("- patriots_vs_vikings_llm_schema.json")


if __name__ == "__main__":
    write_artifacts()
