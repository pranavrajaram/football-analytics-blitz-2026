"""
Builds a styled HTML presentation for the Football Analytics Blitz case.
Output: presentation.html
"""

import csv
import json
from collections import defaultdict

DATA_PATH = "2026_FAB_play_by_play.csv"
OUTPUT_HTML = "presentation.html"

TWO_HIGH = {"cover 2", "cover 4", "man cover 2", "cover 6"}


def is_two_high(cov: str) -> bool:
    return (cov or "").strip().lower() in TWO_HIGH


def to_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


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


def yardline_100(field_side: str, start_yard: str) -> int:
    side = (field_side or "").strip().lower()
    y = to_int(start_yard, 0)
    if side == "own":
        return y
    if side == "oppo":
        return 100 - y
    return 0


def read_rows():
    with open(DATA_PATH, newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames and r.fieldnames[0].startswith("\ufeff"):
            r.fieldnames[0] = r.fieldnames[0].lstrip("\ufeff")
        for row in r:
            if "\ufeffSeason" in row:
                row["Season"] = row.pop("\ufeffSeason")
            yield row


def bar_row(label: str, value: float, max_value: float, suffix: str = "") -> str:
    width = 100 * (value / max_value) if max_value else 0
    return (
        f"<div class='bar-row'>"
        f"<div class='bar-label'>{label}</div>"
        f"<div class='bar'><div class='bar-fill' style='width:{width:.1f}%'></div></div>"
        f"<div class='bar-val'>{value:.2f}{suffix}</div>"
        f"</div>"
    )


def main():
    # Aggregations
    defense_total = defaultdict(int)
    defense_two_high = defaultdict(int)
    defense_pass_epa_sum = defaultdict(float)
    defense_pass_epa_n = defaultdict(int)

    by_down_total = defaultdict(int)
    by_down_two = defaultdict(int)

    pr_epa_sum = defaultdict(float)
    pr_epa_n = defaultdict(int)
    pr_yards_sum = defaultdict(float)
    pr_yards_n = defaultdict(int)
    pr_success_n = defaultdict(int)

    route_epa_sum = defaultdict(float)
    route_epa_n = defaultdict(int)
    route_yards_sum = defaultdict(float)
    route_success_n = defaultdict(int)

    form_epa_sum = defaultdict(float)
    form_epa_n = defaultdict(int)
    form_success_n = defaultdict(int)

    for row in read_rows():
        cov = row.get("CoverageType")
        two = is_two_high(cov)
        def_team = (row.get("DefTeam") or "").strip()
        down = to_int(row.get("Down"))
        if down:
            by_down_total[down] += 1
            if two:
                by_down_two[down] += 1

        if def_team:
            defense_total[def_team] += 1
            if two:
                defense_two_high[def_team] += 1

        if not two:
            continue

        if to_int(row.get("Attempt")) == 1:
            epa = to_float(row.get("EPA"))
            defense_pass_epa_sum[def_team] += epa
            defense_pass_epa_n[def_team] += 1

        event = (row.get("EventType") or "").strip().lower()
        epa = to_float(row.get("EPA"))
        yds = to_float(row.get("YardsOnPlay"))
        if event in ("pass", "rush"):
            pr_epa_sum[event] += epa
            pr_epa_n[event] += 1
            pr_yards_sum[event] += yds
            pr_yards_n[event] += 1
            if epa > 0:
                pr_success_n[event] += 1

        if to_int(row.get("Attempt")) == 1:
            tgt = (row.get("TargetedPlayer") or "").strip()
            if tgt.lower() == "back":
                route = "BACK"
            elif tgt in {"L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"}:
                route = (row.get(tgt) or "").strip() or "UNKNOWN"
            else:
                route = "UNKNOWN"
            route_epa_sum[route] += epa
            route_epa_n[route] += 1
            route_yards_sum[route] += yds
            if epa > 0:
                route_success_n[route] += 1

        form = (row.get("ReceiverAlignment") or "").strip()
        if form:
            form_epa_sum[form] += epa
            form_epa_n[form] += 1
            if epa > 0:
                form_success_n[form] += 1

    # Team two-high usage rate
    usage = []
    for team, total in defense_total.items():
        if total < 50:
            continue
        rate = defense_two_high[team] / total if total else 0
        usage.append((team, rate, defense_two_high[team], total))
    usage_sorted = sorted(usage, key=lambda x: x[1], reverse=True)

    # Defense effectiveness (lower EPA allowed on pass vs two-high)
    eff = []
    for team, n in defense_pass_epa_n.items():
        if n < 40:
            continue
        avg = defense_pass_epa_sum[team] / n
        eff.append((team, avg, n))
    eff_sorted = sorted(eff, key=lambda x: x[1])

    # Down rates
    by_down_rate = []
    for d in sorted(by_down_total):
        rate = by_down_two[d] / by_down_total[d]
        by_down_rate.append((d, rate, by_down_two[d], by_down_total[d]))

    # Pass vs rush
    pr_stats = {}
    for k in ["pass", "rush"]:
        if pr_epa_n[k] > 0:
            pr_stats[k] = {
                "epa": pr_epa_sum[k] / pr_epa_n[k],
                "yards": pr_yards_sum[k] / pr_yards_n[k],
                "n": pr_epa_n[k],
                "success": pr_success_n[k] / pr_epa_n[k],
            }

    # Route top
    routes = []
    for r, n in route_epa_n.items():
        if n < 40:
            continue
        routes.append(
            (
                r,
                route_epa_sum[r] / n,
                route_yards_sum[r] / n,
                n,
                route_success_n[r] / n,
            )
        )
    routes_sorted = sorted(routes, key=lambda x: x[1], reverse=True)[:6]

    # Formation top
    forms = []
    for f, n in form_epa_n.items():
        if n < 80:
            continue
        forms.append((f, form_epa_sum[f] / n, n, form_success_n[f] / n))
    forms_sorted = sorted(forms, key=lambda x: x[1], reverse=True)[:6]

    # Load LLM artifacts
    play_report = json.load(open("patriots_vs_vikings_play_report.json"))
    llm_play = play_report["play"]
    summary = json.load(open("patriots_vs_vikings_summary.json"))
    # Build slide content
    usage_top = usage_sorted[:5]
    usage_bottom = usage_sorted[-5:][::-1]
    max_usage = usage_top[0][1] if usage_top else 1

    usage_top_rows = "\n".join(
        [bar_row(t, r * 100, max_usage * 100, "%") for t, r, _, _ in usage_top]
    )
    usage_bottom_rows = "\n".join(
        [
            bar_row(t, r * 100, max_usage * 100, "%")
            for t, r, _, _ in usage_bottom
        ]
    )

    down_max = max([r for _, r, _, _ in by_down_rate] or [1])
    down_rows = "\n".join(
        [
            bar_row(f"Down {d}", r * 100, down_max * 100, "%")
            for d, r, _, _ in by_down_rate
        ]
    )

    best_def = "\n".join(
        [
            f"<li><strong>{t}</strong> — {avg:+.3f} EPA (n={n})</li>"
            for t, avg, n in eff_sorted[:5]
        ]
    )
    worst_def = "\n".join(
        [
            f"<li><strong>{t}</strong> — {avg:+.3f} EPA (n={n})</li>"
            for t, avg, n in eff_sorted[-5:][::-1]
        ]
    )

    pass_tile = pr_stats.get("pass", {"epa": 0, "yards": 0, "n": 0, "success": 0})
    rush_tile = pr_stats.get("rush", {"epa": 0, "yards": 0, "n": 0, "success": 0})

    routes_rows = "\n".join(
        [
            f"<tr><td>{r}</td><td>{epa:+.3f}</td><td>{yds:.1f}</td><td>{sr*100:.1f}%</td><td>{n}</td></tr>"
            for r, epa, yds, n, sr in routes_sorted
        ]
    )

    forms_rows = "\n".join(
        [
            f"<tr><td>{f}</td><td>{epa:+.3f}</td><td>{sr*100:.1f}%</td><td>{n}</td></tr>"
            for f, epa, n, sr in forms_sorted
        ]
    )

    vik_personnel = (
        summary["top_vikings_two_high_personnel"][0][0]
        if summary["top_vikings_two_high_personnel"]
        else "UNKNOWN"
    )
    vik_covs = summary["top_vikings_two_high_coverages"]
    vik_covs_rows = "\n".join([f"<li>{name}: {count}</li>" for name, count in vik_covs])

    route_items = []
    for k in ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]:
        route_items.append(
            f"<tr><td>{k}</td><td>{llm_play['routes'].get(k, '')}</td></tr>"
        )
    route_table = "\n".join(route_items)

    predict = play_report["predictability_model"]
    expected = play_report["expected_yards"]

    html = f"""
<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<title>Football Analytics Blitz — Two-High Offense Strategy</title>
<style>
:root {{
  --ink: #1c1b1a;
  --forest: #173f35;
  --clay: #d96b27;
  --sand: #f4eee3;
  --stone: #7a726a;
  --mist: #f9f6f0;
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family: 'Avenir Next', 'Helvetica Neue', Arial, sans-serif; color: var(--ink); background: var(--mist); }}
.slide {{ width: 13.333in; height: 7.5in; padding: 0.55in 0.65in; page-break-after: always; background: var(--mist); position: relative; overflow: hidden; }}
.h1 {{ font-size: 40px; font-weight: 700; letter-spacing: -0.5px; margin: 0 0 10px 0; color: var(--forest); }}
.h2 {{ font-size: 24px; font-weight: 700; margin: 8px 0 14px 0; color: var(--forest); }}
.h3 {{ font-size: 18px; font-weight: 700; margin: 12px 0 8px 0; color: var(--forest); }}
.sub {{ font-size: 16px; color: var(--stone); max-width: 900px; }}
.row {{ display: flex; gap: 16px; }}
.col {{ flex: 1; }}
.card {{ background: white; border: 1px solid #e8e0d4; border-radius: 14px; padding: 14px 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }}
.badge {{ display:inline-block; background: var(--forest); color: white; padding: 4px 10px; font-size: 12px; border-radius: 999px; letter-spacing: 0.3px; }}
.bullet {{ margin: 6px 0 0 16px; font-size: 15px; }}
.bullet li {{ margin: 6px 0; }}
.bar-row {{ display:flex; align-items:center; gap:10px; margin: 6px 0; }}
.bar-label {{ width: 80px; font-weight: 600; font-size: 13px; color: var(--forest); }}
.bar {{ flex:1; height: 10px; background: #efe7d8; border-radius: 999px; overflow: hidden; }}
.bar-fill {{ height: 100%; background: linear-gradient(90deg, var(--forest), var(--clay)); }}
.bar-val {{ width: 70px; text-align:right; font-variant-numeric: tabular-nums; font-size: 12px; color: var(--stone); }}
.table {{ width:100%; border-collapse: collapse; font-size: 13px; }}
.table th, .table td {{ border-bottom: 1px solid #ece4d7; padding: 6px 4px; text-align: left; }}
.table th {{ color: var(--forest); font-weight: 700; }}
.kpi {{ display:flex; flex-direction:column; gap:2px; }}
.kpi .value {{ font-size: 26px; font-weight: 700; color: var(--clay); }}
.kpi .label {{ font-size: 12px; color: var(--stone); text-transform: uppercase; letter-spacing: 0.6px; }}
.footer {{ position:absolute; bottom: 12px; right: 18px; font-size: 11px; color: #9a8f83; }}
.section-tag {{ font-size: 12px; color: var(--stone); text-transform: uppercase; letter-spacing: 1.4px; }}
.title-grid {{ display:grid; grid-template-columns: 1.1fr 0.9fr; gap: 16px; align-items: center; }}
.route-table td:first-child {{ width: 40px; font-weight: 700; color: var(--forest); }}
.route-table td {{ vertical-align: top; }}
.muted {{ opacity: 0.65; }}
.placeholder {{ border: 2px dashed #c8bdad; background: #fffdf8; border-radius: 12px; height: 3.6in; display:flex; align-items:center; justify-content:center; color: #9a8f83; font-size: 14px; text-align:center; padding: 12px; }}
.note {{ font-size: 12px; color: #9a8f83; }}
.flow {{ display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; align-items: center; }}
.flow .node {{ background: white; border: 1px solid #e8e0d4; border-radius: 12px; padding: 10px 12px; font-size: 13px; box-shadow: 0 1px 4px rgba(0,0,0,0.04); min-height: 66px; display:flex; align-items:center; justify-content:center; text-align:center; }}
.flow .arrow {{ text-align:center; font-size: 18px; color: var(--stone); }}
.callout {{ border-left: 4px solid var(--clay); padding-left: 10px; color: var(--stone); font-size: 14px; }}
.play-anim {{ border: 1px solid #e8e0d4; border-radius: 12px; padding: 8px; background: #ffffff; }}
.play-anim svg {{ width: 100%; height: auto; display:block; }}
.dash {{ stroke-dasharray: 6 6; }}
.route-line {{ stroke: #d96b27; stroke-width: 3; fill: none; }}
.route-line.alt {{ stroke: #173f35; }}
.route-line.grey {{ stroke: #7a726a; }}
.player {{ fill: #173f35; }}
.defender {{ fill: #7a726a; }}
.ball {{ fill: #d96b27; }}
.animate-route {{ stroke-dasharray: 300; stroke-dashoffset: 300; animation: draw 3.5s ease forwards; }}
@keyframes draw {{ to {{ stroke-dashoffset: 0; }} }}

@media print {{
  body {{ background: white; }}
  .slide {{ box-shadow: none; }}
}}
</style>
</head>
<body>

<section class='slide'>
  <div class='title-grid'>
    <div>
      <div class='badge'>Football Analytics Blitz — 2026</div>
      <div class='h1'>Defeating the Two-High</div>
      <div class='sub'>Strategic offensive adaptations, data-driven counters, and a concrete play design vs Minnesota's two-high shell.</div>
      <div class='h3'>Team: Patriots vs Vikings</div>
    </div>
    <div class='card'>
      <div class='h2'>What We Deliver</div>
      <ul class='bullet'>
        <li>League-wide two-high usage + effectiveness</li>
        <li>Offensive counter-strategies by route & formation</li>
        <li>Coaching game plan + specific play design</li>
      </ul>
    </div>
  </div>
  <div class='footer'>Data: 2026_FAB_play_by_play</div>
</section>

<section class='slide'>
  <div class='section-tag'>Methodology</div>
  <div class='h1'>Approach & Workflow</div>
  <div class='row'>
    <div class='col card'>
      <div class='h2'>Part 1: Deployment & Trends</div>
      <ol class='bullet'>
        <li>Filter to two-high shells (Cover 2/4/6, Man Cover 2).</li>
        <li>Compute usage rates by team + by down/field zone.</li>
        <li>Rank defenses by pass EPA allowed vs two-high.</li>
      </ol>
    </div>
    <div class='col card'>
      <div class='h2'>Part 2: Offensive Counters</div>
      <ol class='bullet'>
        <li>Compare pass vs rush EPA + success rate vs two-high.</li>
        <li>Rank routes and formations by EPA + success rate.</li>
        <li>Translate to actionable design rules (safety binds, honey-hole).</li>
      </ol>
    </div>
    <div class='col card'>
      <div class='h2'>Part 3: Game Plan & Play</div>
      <ol class='bullet'>
        <li>Extract Vikings’ most common two-high personnel + coverages.</li>
        <li>Generate a play from inputs (down, distance, field position, personnel).</li>
        <li>Score play with EY + predictability model.</li>
      </ol>
    </div>
  </div>
</section>

<section class='slide'>
  <div class='section-tag'>Methodology — Part 3</div>
  <div class='h1'>Decision Loop: More Than “Just Generate a Play”</div>
  <div class='row'>
    <div class='col card'>
      <div class='h2'>Reverse Prompting Motivation</div>
      <ul class='bullet'>
        <li>We include both high-EPA concepts and noisy counterexamples so the LLM learns the <em>shape</em> of a good two-high beater.</li>
        <li>Noise forces robustness: the model must explain <em>why</em> a concept wins, not just mimic a template.</li>
      </ul>
      <div class='callout'>Chess analogy: the best move isn’t always the most common move. We bias toward <em>strong but surprising</em> plays that still score well by EPA and Predictability.</div>
    </div>
    <div class='col card'>
      <div class='h2'>Predictability Model (LSTM)</div>
      <ul class='bullet'>
        <li>LSTM uses drive history + current game state to predict formation probability.</li>
        <li>Predictability Score = P(formation | context). Lower = more surprising.</li>
        <li>This gives a measurable “deception” axis, not just intuition.</li>
      </ul>
      <div class='h2' style='margin-top:10px;'>EDA-Powered Feedback</div>
      <ul class='bullet'>
        <li>We score EY and predictability after generation.</li>
        <li>We iterate until the concept balances yardage + disguise.</li>
      </ul>
    </div>
  </div>
  <div class='row' style='margin-top:12px;'>
    <div class='col card'>
      <div class='h2'>Pipeline Flow</div>
      <div class='flow'>
        <div class='node'>EDA + Filters<br><span class='note'>two-high + situation</span></div>
        <div class='arrow'>→</div>
        <div class='node'>LLM Play Proposal<br><span class='note'>formation + L1–R1 routes</span></div>
        <div class='arrow'>→</div>
        <div class='node'>EY & Predictability<br><span class='note'>LSTM P(formation)</span></div>
        <div class='arrow'>→</div>
        <div class='node'>Coach Review / Iterate<br><span class='note'>keep strong & surprising</span></div>
      </div>
    </div>
  </div>
</section>

<section class='slide'>
  <div class='section-tag'>Part 1</div>
  <div class='h1'>When Defenses Call Two-High</div>
  <div class='row'>
    <div class='col card'>
      <div class='h2'>Two-High Rate by Down</div>
      {down_rows}
    </div>
    <div class='col card'>
      <div class='h2'>Highest Usage Teams</div>
      {usage_top_rows}
      <div class='h2' style='margin-top:16px;'>Lowest Usage Teams</div>
      {usage_bottom_rows}
    </div>
  </div>
</section>

<section class='slide'>
  <div class='section-tag'>Part 1</div>
  <div class='h1'>Two-High Effectiveness by Team</div>
  <div class='row'>
    <div class='col card'>
      <div class='h2'>Most Effective (Lowest Pass EPA Allowed)</div>
      <ul class='bullet'>
        {best_def}
      </ul>
    </div>
    <div class='col card'>
      <div class='h2'>Least Effective (Highest Pass EPA Allowed)</div>
      <ul class='bullet'>
        {worst_def}
      </ul>
    </div>
  </div>
  <div class='sub' style='margin-top:10px;'>Interpretation: two-high usage varies widely by team; effectiveness depends on the pass rush + disguise quality, not just shell frequency.</div>
</section>

<section class='slide'>
  <div class='section-tag'>Part 2</div>
  <div class='h1'>What Works vs Two-High</div>
  <div class='row'>
    <div class='col card'>
      <div class='h2'>Top Route Concepts</div>
      <table class='table'>
        <thead><tr><th>Route</th><th>EPA</th><th>Yards</th><th>Success</th><th>n</th></tr></thead>
        <tbody>
          {routes_rows}
        </tbody>
      </table>
    </div>
    <div class='col card'>
      <div class='h2'>Top Formations</div>
      <table class='table'>
        <thead><tr><th>Formation</th><th>EPA</th><th>Success</th><th>n</th></tr></thead>
        <tbody>
          {forms_rows}
        </tbody>
      </table>
    </div>
  </div>
  <div class='row' style='margin-top:12px;'>
    <div class='col card muted'>
      <div class='h3'>Pass vs Rush (Two-High)</div>
      <div class='row'>
        <div class='col'>
          <div class='kpi'><div class='label'>Pass EPA</div><div class='value'>{pass_tile['epa']:+.3f}</div><div class='label'>Success {pass_tile['success']*100:.1f}% • Yards {pass_tile['yards']:.1f} (n={pass_tile['n']})</div></div>
        </div>
        <div class='col'>
          <div class='kpi'><div class='label'>Rush EPA</div><div class='value'>{rush_tile['epa']:+.3f}</div><div class='label'>Success {rush_tile['success']*100:.1f}% • Yards {rush_tile['yards']:.1f} (n={rush_tile['n']})</div></div>
        </div>
      </div>
    </div>
    <div class='col card muted'>
      <div class='h3'>Design Rules</div>
      <ul class='bullet'>
        <li>Clearout (post/vertical) to open honey-hole (15–22 yards).</li>
        <li>Create high-low or inside-out bind on the field safety.</li>
        <li>Avoid horizontal clutter that invites rally-and-tackle.</li>
      </ul>
    </div>
  </div>
</section>

<section class='slide'>
  <div class='section-tag'>Part 3</div>
  <div class='h1'>Matchup Context: Patriots vs Vikings</div>
  <div class='row'>
    <div class='col card'>
      <div class='h2'>Vikings Two-High Tendencies</div>
      <ul class='bullet'>
        <li>Most common two-high personnel: <strong>{vik_personnel}</strong></li>
        <li>Coverage mix: {", ".join([f"{name} ({count})" for name, count in vik_covs])}</li>
      </ul>
    </div>
    <div class='col card'>
      <div class='h2'>Play Design Inputs</div>
      <ul class='bullet'>
        <li>Down & Distance: 1st & 10</li>
        <li>Field Position: Own 35</li>
        <li>Defensive Personnel: {vik_personnel}</li>
      </ul>
      <div class='sub' style='margin-top:10px;'>Goal: keep two-high on the field while attacking the intermediate honey-hole.</div>
    </div>
  </div>
</section>

<section class='slide'>
  <div class='section-tag'>Part 3</div>
  <div class='h1'>Designed Play: LLM Output</div>
  <div class='row'>
    <div class='col card'>
      <div class='h2'>Formation</div>
      <div class='sub'>{llm_play['formation']}</div>
      <div class='h2' style='margin-top:12px;'>Routes (L1–R1)</div>
      <table class='table route-table'>
        <tbody>
          {route_table}
        </tbody>
      </table>
    </div>
    <div class='col card'>
      <div class='h2'>Play Diagram (Insert)</div>
      <!-- Best rendering: export a wide 16:9 play diagram PNG at ~2200px width, then replace this div with <img src='play_diagram.png' style='width:100%; height:auto;'> -->
      <div class='placeholder'>Placeholder for play diagram image (16:9). Replace with a PNG/SVG of the drawn-up play.</div>
      <div class='note' style='margin-top:6px;'>Tip: Use a transparent-background PNG or SVG for crisp lines in PDF.</div>
      <div class='h2' style='margin-top:10px;'>Rationale & Scores</div>
      <div class='sub'>{llm_play['rationale']}</div>
      <ul class='bullet'>
        <li>Expected Yards: <strong>{expected['expected_yards']:.2f}</strong> (n={expected['sample_n']}, {expected['level']})</li>
        <li>P-score (formation surprise): <strong>{predict['p_score']:.3f}</strong></li>
        <li>Relative predictability: <strong>{predict['relative_predictability']:+.3f}</strong></li>
      </ul>
    </div>
  </div>
</section>

<section class='slide'>
  <div class='section-tag'>Part 3</div>
  <div class='h1'>Play Animation (Draft)</div>
  <div class='row'>
    <div class='col card'>
      <div class='h2'>Animated Route Sketch</div>
      <div class='play-anim'>
        <svg viewBox="0 0 900 480" xmlns="http://www.w3.org/2000/svg">
          <!-- Field -->
          <rect x="20" y="20" width="860" height="440" rx="18" fill="#f7f2e8" stroke="#d8cdbf"/>
          <line x1="450" y1="20" x2="450" y2="460" stroke="#e1d5c7" stroke-width="2" stroke-dasharray="6 6"/>
          <line x1="20" y1="240" x2="880" y2="240" stroke="#e1d5c7" stroke-width="2" stroke-dasharray="6 6"/>

          <!-- Offense (bottom) -->
          <circle class="player" cx="260" cy="420" r="7"/>
          <circle class="player" cx="320" cy="420" r="7"/>
          <circle class="player" cx="380" cy="420" r="7"/>
          <circle class="player" cx="520" cy="420" r="7"/>
          <circle class="player" cx="580" cy="420" r="7"/>
          <circle class="player" cx="640" cy="420" r="7"/>

          <!-- Defense (two-high shell, top) -->
          <circle class="defender" cx="260" cy="120" r="6"/>
          <circle class="defender" cx="450" cy="120" r="6"/>
          <circle class="defender" cx="640" cy="120" r="6"/>
          <circle class="defender" cx="350" cy="160" r="6"/>
          <circle class="defender" cx="550" cy="160" r="6"/>

          <!-- Routes (animated) -->
          <path class="route-line animate-route" d="M260 420 C260 320 250 240 230 160" />
          <path class="route-line alt animate-route" d="M520 420 C520 340 520 300 520 240 C520 200 540 180 570 160" />
          <path class="route-line grey animate-route" d="M640 420 C660 360 700 300 760 240" />
          <path class="route-line animate-route" d="M320 420 C320 380 320 360 320 340" />
        </svg>
      </div>
      <div class='note' style='margin-top:8px;'>
        This is a lightweight SVG animation for screen preview. For the final PDF, replace with a static PNG/SVG play diagram.
      </div>
    </div>
    <div class='col card'>
      <div class='h2'>How to Render Final Diagram</div>
      <ul class='bullet'>
        <li>Export a 16:9 field diagram as PNG or SVG (2200px wide recommended).</li>
        <li>Prefer SVG for sharp vectors in PDF; PNG with transparent background also works.</li>
        <li>Replace the placeholder on the prior slide with an <code>&lt;img&gt;</code> tag.</li>
      </ul>
      <div class='callout'>We can generate a final static diagram from the routes once you confirm exact route landmarks.</div>
    </div>
  </div>
</section>

<section class='slide'>
  <div class='section-tag'>Wrap</div>
  <div class='h1'>Key Takeaways</div>
  <div class='row'>
    <div class='col card'>
      <div class='h2'>What We Learned</div>
      <ul class='bullet'>
        <li>Two-high frequency is team- and situation-dependent (not uniform).</li>
        <li>Vertical clearouts + intermediate targets drive higher EPA.</li>
        <li>Predictability suppression is measurable using formation probability.</li>
      </ul>
    </div>
    <div class='col card'>
      <div class='h2'>How We Apply It</div>
      <ul class='bullet'>
        <li>Design plays that force safety binds (high-low / inside-out).</li>
        <li>Use 11 personnel or trips alignments to disguise intent.</li>
        <li>Score each concept by EY and predictability before finalizing.</li>
      </ul>
    </div>
  </div>
  <div class='footer'>Ready for Q&A</div>
</section>

</body>
</html>
"""

    with open(OUTPUT_HTML, "w") as f:
        f.write(html)

    print(f"Wrote {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
