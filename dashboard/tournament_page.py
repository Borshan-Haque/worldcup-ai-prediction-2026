"""
WorldCupAI — Tournament Predictor Page
Flagship page: simulates every fixture of FIFA WC 2026 in chronological order.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from utils.tournament import (
    TournamentPredictor,
    TournamentResult,
    load_fixtures,
    rank_group,
    GROUPS,
)

FLAGS = {
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷",
    "Czech Republic": "🇨🇿", "Canada": "🇨🇦", "Bosnia and Herzegovina": "🇧🇦",
    "Qatar": "🇶🇦", "Switzerland": "🇨🇭", "Brazil": "🇧🇷", "Morocco": "🇲🇦",
    "Haiti": "🇭🇹", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "United States": "🇺🇸",
    "Paraguay": "🇵🇾", "Australia": "🇦🇺", "Turkey": "🇹🇷",
    "Germany": "🇩🇪", "Curaçao": "🇨🇼", "Ivory Coast": "🇨🇮",
    "Ecuador": "🇪🇨", "Netherlands": "🇳🇱", "Japan": "🇯🇵",
    "Sweden": "🇸🇪", "Tunisia": "🇹🇳", "Belgium": "🇧🇪",
    "Egypt": "🇪🇬", "Iran": "🇮🇷", "New Zealand": "🇳🇿",
    "Spain": "🇪🇸", "Cape Verde": "🇨🇻", "Saudi Arabia": "🇸🇦",
    "Uruguay": "🇺🇾", "France": "🇫🇷", "Senegal": "🇸🇳",
    "Iraq": "🇮🇶", "Norway": "🇳🇴", "Argentina": "🇦🇷",
    "Algeria": "🇩🇿", "Austria": "🇦🇹", "Jordan": "🇯🇴",
    "Portugal": "🇵🇹", "DR Congo": "🇨🇩", "Uzbekistan": "🇺🇿",
    "Colombia": "🇨🇴", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Croatia": "🇭🇷",
    "Ghana": "🇬🇭", "Panama": "🇵🇦",
}

def flag(t: str) -> str:
    return FLAGS.get(t, "🏳️")


def clean_html(html_str: str) -> str:
    if not html_str:
        return ""
    return " ".join(line.strip() for line in html_str.splitlines() if line.strip())


# ── Reusable card renderers ───────────────────────────────────────────────

def _match_card(fix, compact: bool = False) -> str:
    """Generate HTML for a single match result card."""
    h, a = fix.home, fix.away
    hw = fix.home_score > fix.away_score
    aw = fix.away_score > fix.home_score
    h_col = "#00d4ff" if hw else ("#ffc72c" if aw else "#6b7a95")
    a_col = "#00d4ff" if aw else ("#ffc72c" if hw else "#6b7a95")

    extra = ""
    if fix.went_to_pens:
        extra = f'<div style="font-size:0.75rem;color:#ffc72c;margin-top:2px;">Pens {fix.pen_home}–{fix.pen_away}</div>'
    elif fix.went_to_et:
        extra = '<div style="font-size:0.75rem;color:#6b7a95;margin-top:2px;">AET</div>'

    reasons_html = ""
    if fix.reasoning and not compact:
        items = "".join(
            f'<div style="font-size:0.72rem;color:#6b7a95;margin-top:2px;">✓ {r}</div>'
            for r in fix.reasoning[:3]
        )
        reasons_html = f'<div style="margin-top:10px;padding-top:10px;border-top:1px solid #1e2d42;">{items}</div>'

    prob_bar = ""
    if not compact:
        prob_bar = f"""
        <div style="display:flex;gap:2px;height:4px;border-radius:2px;overflow:hidden;margin:10px 0 6px 0;">
            <div style="width:{fix.home_win_pct}%;background:#00d4ff;"></div>
            <div style="width:{fix.draw_pct}%;background:#2d3f55;"></div>
            <div style="width:{fix.away_win_pct}%;background:#ffc72c;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.68rem;color:#6b7a95;">
            <span>{fix.home_win_pct}%</span><span>Draw {fix.draw_pct}%</span><span>{fix.away_win_pct}%</span>
        </div>"""

    xg_row = ""
    if not compact:
        xg_row = f'<div style="font-size:0.68rem;color:#6b7a95;text-align:center;margin-top:4px;">xG: {fix.home_xg} — {fix.away_xg}</div>'

    return clean_html(f"""
    <div style="background:#131926;border:1px solid #1e2d42;border-radius:10px;
                padding:{'12px' if compact else '16px'};margin-bottom:10px;position:relative;">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
            <div style="flex:1;text-align:right;">
                <div style="font-size:{'1.3' if compact else '1.6'}rem;">{flag(h)}</div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                            font-size:{'0.85' if compact else '0.95'}rem;color:{h_col};">{h}</div>
            </div>
            <div style="text-align:center;min-width:80px;">
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:{'1.8' if compact else '2.2'}rem;
                            font-weight:800;color:#ffc72c;line-height:1;">
                    {fix.home_score}–{fix.away_score}
                </div>
                {extra}
                {xg_row}
            </div>
            <div style="flex:1;text-align:left;">
                <div style="font-size:{'1.3' if compact else '1.6'}rem;">{flag(a)}</div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                            font-size:{'0.85' if compact else '0.95'}rem;color:{a_col};">{a}</div>
            </div>
        </div>
        {prob_bar}
        {reasons_html}
    </div>""")


def _standings_html(group: str, standings: dict) -> str:
    ranked = rank_group(standings[group])
    rows = ""
    for i, st in enumerate(ranked):
        highlight = "border-left:3px solid #00d4ff;" if i < 2 else (
            "border-left:3px solid #ffc72c;" if i == 2 else ""
        )
        bg = "background:rgba(0,212,255,0.05);" if i < 2 else ""
        rows += f"""
        <div style="display:flex;align-items:center;gap:8px;padding:5px 6px;
                    margin-bottom:3px;border-radius:4px;{highlight}{bg}">
            <span style="width:16px;font-size:0.75rem;color:#6b7a95;">{i+1}</span>
            <span style="font-size:1.1rem;">{flag(st.team)}</span>
            <span style="flex:1;font-weight:600;font-size:0.82rem;overflow:hidden;
                         text-overflow:ellipsis;white-space:nowrap;">{st.team}</span>
            <span style="width:22px;text-align:center;font-size:0.78rem;color:#6b7a95;">{st.played}</span>
            <span style="width:22px;text-align:center;font-size:0.78rem;color:#6b7a95;">{st.won}</span>
            <span style="width:22px;text-align:center;font-size:0.78rem;color:#6b7a95;">{st.drawn}</span>
            <span style="width:22px;text-align:center;font-size:0.78rem;color:#6b7a95;">{st.lost}</span>
            <span style="width:26px;text-align:center;font-size:0.78rem;color:#6b7a95;">{st.gf}</span>
            <span style="width:26px;text-align:center;font-size:0.78rem;color:#6b7a95;">{st.ga}</span>
            <span style="width:30px;text-align:center;font-size:0.78rem;
                         color:{'#00e676' if st.gd>0 else '#ff4444' if st.gd<0 else '#6b7a95'};">
                {st.gd:+d}</span>
            <span style="width:28px;text-align:center;font-family:'Barlow Condensed',sans-serif;
                         font-size:1rem;font-weight:700;color:#ffc72c;">{st.pts}</span>
        </div>"""
    header = """
    <div style="display:flex;align-items:center;gap:8px;padding:3px 6px;margin-bottom:4px;">
        <span style="width:16px;"></span><span style="width:20px;"></span>
        <span style="flex:1;font-size:0.65rem;color:#2d3f55;text-transform:uppercase;letter-spacing:0.1em;">Team</span>
        <span style="width:22px;text-align:center;font-size:0.65rem;color:#2d3f55;">P</span>
        <span style="width:22px;text-align:center;font-size:0.65rem;color:#2d3f55;">W</span>
        <span style="width:22px;text-align:center;font-size:0.65rem;color:#2d3f55;">D</span>
        <span style="width:22px;text-align:center;font-size:0.65rem;color:#2d3f55;">L</span>
        <span style="width:26px;text-align:center;font-size:0.65rem;color:#2d3f55;">GF</span>
        <span style="width:26px;text-align:center;font-size:0.65rem;color:#2d3f55;">GA</span>
        <span style="width:30px;text-align:center;font-size:0.65rem;color:#2d3f55;">GD</span>
        <span style="width:28px;text-align:center;font-size:0.65rem;color:#2d3f55;">PTS</span>
    </div>"""
    return clean_html(f"""
    <div style="background:#0e1420;border:1px solid #1e2d42;border-radius:8px;padding:12px 14px;margin-bottom:12px;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:800;
                    text-transform:uppercase;color:#ffc72c;letter-spacing:0.1em;margin-bottom:8px;">
            Group {group}
        </div>
        {header}{rows}
        <div style="font-size:0.6rem;color:#1e2d42;margin-top:6px;">
            ↑ Top 2 + best 3rd qualify
        </div>
    </div>""")


def _timeline_html(active_stage: str) -> str:
    stages = ["Group Stage", "Round of 32", "Round of 16",
              "Quarter-Finals", "Semi-Finals", "Final", "Champion"]
    items = ""
    reached = False
    for s in stages:
        if s == active_stage:
            reached = True
        color = "#ffc72c" if s == active_stage else ("#00d4ff" if not reached else "#1e2d42")
        weight = "800" if s == active_stage else "600"
        items += f"""
        <div style="display:flex;flex-direction:column;align-items:center;gap:4px;">
            <div style="width:10px;height:10px;border-radius:50%;background:{color};
                        {'box-shadow:0 0 8px #ffc72c;' if s==active_stage else ''}"></div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.7rem;
                        font-weight:{weight};color:{color};text-transform:uppercase;
                        letter-spacing:0.06em;text-align:center;max-width:60px;">{s}</div>
        </div>
        <div style="flex:1;height:2px;background:{'#ffc72c' if not reached or s==active_stage else '#1e2d42'};
                    margin-bottom:14px;min-width:16px;max-width:40px;"></div>"""
    return clean_html(f"""
    <div style="display:flex;align-items:flex-start;justify-content:center;
                padding:12px 0;overflow-x:auto;">
        {items.rsplit('<div style="flex:1', 1)[0]}
    </div>""")


# ── Main page renderer ────────────────────────────────────────────────────

def render_tournament_predictor_page(engine):
    st.markdown('<p class="hero-title">Tournament Predictor</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#6b7a95;font-size:1rem;margin-bottom:24px;">'
        'Predict every match of the FIFA World Cup 2026 using AI — '
        'fixture by fixture, in official tournament order.</p>',
        unsafe_allow_html=True,
    )

    # ── Overview cards ────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, "48", "Teams"),
        (c2, "104", "Total Matches"),
        (c3, "12", "Groups"),
        (c4, "6", "Knockout Rounds"),
    ]:
        col.markdown(clean_html(f"""
        <div class="card-sm" style="text-align:center;">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:2.2rem;
                        font-weight:800;color:#ffc72c;">{val}</div>
            <div class="stat-label">{lbl}</div>
        </div>"""), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Action button ─────────────────────────────────────────────────
    col_btn, col_clear = st.columns([2, 1])
    run_clicked   = col_btn.button("🚀  Predict Entire Tournament", use_container_width=True)
    clear_clicked = col_clear.button("🔄  Reset", use_container_width=True)

    if clear_clicked:
        for k in ["tp_result", "tp_stage"]:
            st.session_state.pop(k, None)
        st.rerun()

    # ── Run simulation ────────────────────────────────────────────────
    if run_clicked and "tp_result" not in st.session_state:
        predictor = TournamentPredictor(engine)
        bar = st.progress(0, text="Simulating Group Stage…")
        result = predictor.predict_tournament()
        bar.progress(1.0, text="Complete!")
        time.sleep(0.4)
        bar.empty()
        st.session_state["tp_result"] = result
        st.session_state["tp_stage"] = "Final"
        st.rerun()

    result: TournamentResult | None = st.session_state.get("tp_result")

    if result is None:
        st.markdown(clean_html("""
        <div style="background:#0e1420;border:1px dashed #1e2d42;border-radius:12px;
                    padding:48px;text-align:center;margin-top:16px;">
            <div style="font-size:3rem;margin-bottom:12px;">⚽</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.4rem;
                        font-weight:700;color:#6b7a95;text-transform:uppercase;
                        letter-spacing:0.1em;">
                Click the button above to simulate the full tournament
            </div>
        </div>"""), unsafe_allow_html=True)
        return

    # ── Champion banner ───────────────────────────────────────────────
    st.markdown(_timeline_html("Final"), unsafe_allow_html=True)
    st.markdown(clean_html(f"""
    <div style="background:linear-gradient(135deg,rgba(255,199,44,0.12),rgba(0,212,255,0.08));
                border:1px solid #ffc72c;border-radius:16px;padding:28px;text-align:center;
                margin-bottom:24px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;
                    background:linear-gradient(90deg,#ffc72c,#00d4ff,#ffc72c);"></div>
        <div style="font-size:0.75rem;color:#ffc72c;text-transform:uppercase;
                    letter-spacing:0.25em;font-weight:700;margin-bottom:8px;">
            🏆 FIFA World Cup 2026 Champion
        </div>
        <div style="font-size:4.5rem;line-height:1;">{flag(result.champion)}</div>
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:2.8rem;
                    font-weight:800;text-transform:uppercase;letter-spacing:0.05em;
                    margin:4px 0 8px 0;">{result.champion}</div>
        <div style="font-size:0.85rem;color:#6b7a95;">
            Runner-Up: {flag(result.runner_up)} {result.runner_up}
        </div>
    </div>"""), unsafe_allow_html=True)

    # ── Final match ───────────────────────────────────────────────────
    final_fix = next((f for f in result.fixtures if f.round == "Final"), None)
    if final_fix:
        st.markdown('<p class="section-title">⚽ The Final</p>', unsafe_allow_html=True)
        st.markdown(_match_card(final_fix, compact=False), unsafe_allow_html=True)

    # ── Tournament stats ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">Tournament Statistics</p>', unsafe_allow_html=True)
    s = result.stats
    stat_cols = st.columns(4)
    for col, val, lbl in [
        (stat_cols[0], s.get("total_goals", 0),      "Total Goals"),
        (stat_cols[1], s.get("goals_per_match", 0),  "Goals / Match"),
        (stat_cols[2], s.get("total_matches", 0),    "Matches Played"),
        (stat_cols[3], s.get("top_scorer_team_goals", 0), "Top Team Goals"),
    ]:
        col.markdown(clean_html(f"""
        <div class="card-sm" style="text-align:center;">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;
                        font-weight:800;color:#00d4ff;">{val}</div>
            <div class="stat-label">{lbl}</div>
        </div>"""), unsafe_allow_html=True)

    info_cols = st.columns(3)
    for col, lbl, val in [
        (info_cols[0], "🥇 Top Scoring Team",    f"{flag(s.get('top_scorer_team',''))} {s.get('top_scorer_team','')} ({s.get('top_scorer_team_goals',0)} goals)"),
        (info_cols[1], "🛡️ Best Defence",         f"{flag(s.get('best_defense_team',''))} {s.get('best_defense_team','')} ({s.get('best_defense_goals_conceded',0)} conceded)"),
        (info_cols[2], "😮 Biggest Upset",        s.get("biggest_upset", "None")),
    ]:
        col.markdown(clean_html(f"""
        <div class="card-sm" style="margin-top:8px;">
            <div class="stat-label" style="margin-bottom:4px;">{lbl}</div>
            <div style="font-weight:600;font-size:0.9rem;">{val}</div>
        </div>"""), unsafe_allow_html=True)

    info_cols2 = st.columns(2)
    info_cols2[0].markdown(clean_html(f"""
    <div class="card-sm" style="margin-top:8px;">
        <div class="stat-label">🔥 Highest Scoring Match</div>
        <div style="font-weight:600;font-size:0.9rem;margin-top:4px;">
            {s.get('highest_scoring_match','')}
        </div>
    </div>"""), unsafe_allow_html=True)

    # ── Knockout rounds ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">Knockout Stage</p>', unsafe_allow_html=True)

    round_order = ["Semi-Finals", "Quarter-Finals", "Round of 16", "Round of 32"]
    for rnd in round_order:
        rnd_fixes = [f for f in result.fixtures if f.round == rnd and f.played]
        if not rnd_fixes:
            continue
        st.markdown(clean_html(f"""
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
                    font-weight:700;text-transform:uppercase;letter-spacing:0.12em;
                    color:#6b7a95;margin:16px 0 10px 0;">{rnd}</div>"""),
                    unsafe_allow_html=True)
        cols_per_row = 4 if rnd == "Round of 32" else (2 if rnd in ("Round of 16", "Quarter-Finals") else 2)
        for start in range(0, len(rnd_fixes), cols_per_row):
            chunk = rnd_fixes[start:start + cols_per_row]
            cols = st.columns(len(chunk))
            for col, fix in zip(cols, chunk):
                col.markdown(_match_card(fix, compact=True), unsafe_allow_html=True)

    # Bronze final
    bronze = next((f for f in result.fixtures if f.round == "Bronze Final"), None)
    if bronze and bronze.played:
        st.markdown(clean_html("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
                        font-weight:700;text-transform:uppercase;letter-spacing:0.12em;
                        color:#6b7a95;margin:16px 0 10px 0;">🥉 Third Place</div>"""),
                    unsafe_allow_html=True)
        col_b, _ = st.columns([1, 2])
        col_b.markdown(_match_card(bronze, compact=True), unsafe_allow_html=True)

    # ── Group stage results ───────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">Group Stage Results & Standings</p>', unsafe_allow_html=True)

    group_letters = sorted(GROUPS.keys())
    for row_start in range(0, len(group_letters), 3):
        chunk = group_letters[row_start:row_start + 3]
        cols = st.columns(3)
        for col, g in zip(cols, chunk):
            with col:
                col.markdown(_standings_html(g, result.standings), unsafe_allow_html=True)
                g_fixes = [f for f in result.fixtures
                           if f.group == g and f.round == "Group Stage" and f.played]
                for fix in g_fixes:
                    col.markdown(_match_card(fix, compact=True), unsafe_allow_html=True)

    # ── Qualification table ───────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">Qualification Status</p>', unsafe_allow_html=True)

    q_rows = {"Group Winner ✅": [], "Runner-Up ✅": [], "Best 3rd ✅": [], "Eliminated ❌": []}
    for team, status in result.qualified.items():
        if status in q_rows:
            q_rows[status].append(team)

    q_cols = st.columns(4)
    labels = list(q_rows.keys())
    colors = ["#00d4ff", "#ffc72c", "#00e676", "#ff4444"]
    for col, lbl, color in zip(q_cols, labels, colors):
        teams = q_rows[lbl]
        items = "".join(
            f'<div style="padding:4px 0;font-size:0.82rem;">{flag(t)} {t}</div>'
            for t in sorted(teams)
        )
        col.markdown(clean_html(f"""
        <div style="background:#0e1420;border:1px solid #1e2d42;border-radius:8px;
                    padding:14px;max-height:320px;overflow-y:auto;">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.85rem;
                        font-weight:700;color:{color};text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:8px;">
                {lbl} ({len(teams)})
            </div>
            {items}
        </div>"""), unsafe_allow_html=True)

    # ── Matchday breakdown (expandable) ──────────────────────────────
    st.markdown("---")
    for md in [1, 2, 3]:
        md_fixes = [f for f in result.fixtures
                    if f.matchday == md and f.played]
        if not md_fixes:
            continue
        with st.expander(f"📅  Matchday {md} — {len(md_fixes)} matches", expanded=(md == 1)):
            groups_this_md = sorted({f.group for f in md_fixes})
            for g in groups_this_md:
                st.markdown(f"**Group {g}**")
                g_fixes = [f for f in md_fixes if f.group == g]
                cols = st.columns(min(2, len(g_fixes)))
                for i, fix in enumerate(g_fixes):
                    cols[i % len(cols)].markdown(_match_card(fix, compact=False), unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">Export Prediction Report</p>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    predictor = TournamentPredictor(engine)
    
    e1.download_button(
        "📥  Download CSV",
        data=predictor.export_prediction_report(result, "csv"),
        file_name="worldcup_2026_prediction.csv",
        mime="text/csv",
        use_container_width=True,
    )
    e2.download_button(
        "📥  Download JSON",
        data=predictor.export_prediction_report(result, "json"),
        file_name="worldcup_2026_prediction.json",
        mime="application/json",
        use_container_width=True,
    )
    e3.download_button(
        "📥  Download Report (TXT)",
        data=predictor.export_prediction_report(result, "txt"),
        file_name="worldcup_2026_report.txt",
        mime="text/plain",
        use_container_width=True,
    )
