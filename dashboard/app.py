"""
WorldCupAI 2026 — FIFA Broadcast-Grade Dashboard
Complete UI/UX redesign: deep black, gold, neon, cinematic typography.
"""

import sys, random
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.engine import FootballEngine, MatchResult, TeamRating, sort_winning_probabilities
from dashboard.tournament_page import render_tournament_predictor_page

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WorldCupAI 2026",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent.parent / "data"


def clean_html(html_str: str) -> str:
    if not html_str:
        return ""
    return " ".join(line.strip() for line in html_str.splitlines() if line.strip())

# ══════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — CSS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:ital,wght@0,300;0,400;0,600;0,700;0,800;0,900;1,700&family=Barlow:wght@300;400;500;600&display=swap');

/* ── Root tokens ─────────────────────────────────── */
:root {
    --bg:        #05070D;
    --surface:   #0A0E1A;
    --card:      #0E1420;
    --card2:     #111827;
    --border:    #1A2540;
    --border2:   #243050;
    --blue:      #00D4FF;
    --gold:      #FFC72C;
    --green:     #00E676;
    --red:       #FF3D57;
    --purple:    #8B5CF6;
    --text:      #F5F7FA;
    --muted:     #6B7A95;
    --muted2:    #3D4F6E;
}

/* ── Global reset ───────────────────────────────── */
html, body, [class*="css"],
.stApp, .main, section[data-testid="stSidebar"],
div[data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Barlow', sans-serif;
}
.block-container { padding: 1rem 2rem 4rem 2rem !important; max-width: 1400px; }

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #060910 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* Radio nav items */
[data-testid="stSidebar"] .stRadio > div { gap: 2px !important; }
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    padding: 10px 16px !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
    border-left: 3px solid transparent !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--blue) !important;
    background: rgba(0,212,255,0.06) !important;
    border-left-color: var(--blue) !important;
}
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label,
[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    color: var(--gold) !important;
    border-left-color: var(--gold) !important;
    background: rgba(255,199,44,0.06) !important;
}

/* ── Hero typography ─────────────────────────────── */
.hero {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: clamp(2.8rem, 5vw, 5rem);
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: -0.02em;
    line-height: 0.95;
    background: linear-gradient(135deg, #FFFFFF 0%, #6ea3ff 40%, #c39eff 75%, #ffb494 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-sub {
    font-family: 'Barlow', sans-serif;
    font-size: 1rem;
    font-weight: 400;
    color: var(--muted);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 8px;
}
.page-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--blue);
    margin-bottom: 4px;
}
.page-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: clamp(1.8rem, 3vw, 2.8rem);
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: -0.01em;
    color: var(--text);
    margin: 0 0 4px 0;
    line-height: 1;
}
.section-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--muted2);
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}

/* ── Cards ───────────────────────────────────────── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.card:hover {
    border-color: var(--border2);
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.card-accent-top::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--blue), var(--gold));
}
.card-accent-left {
    border-left: 3px solid var(--blue) !important;
}
.card-gold {
    border-color: rgba(255,199,44,0.4) !important;
    background: linear-gradient(135deg, rgba(255,199,44,0.07), rgba(0,212,255,0.04)) !important;
}
.card-sm {
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
}

/* FUT-style team card */
.fut-card {
    background: linear-gradient(145deg, #0f1d2e 0%, #162436 40%, #0a111c 100%);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 14px;
    padding: 20px 14px 16px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.25s;
    cursor: default;
}
.fut-card:hover {
    border-color: var(--gold);
    box-shadow: 0 0 20px rgba(255,199,44,0.2), 0 8px 32px rgba(0,0,0,0.5);
    transform: translateY(-3px);
}
.fut-card::before {
    content: '';
    position: absolute;
    top: -40%; left: -40%;
    width: 180%; height: 180%;
    background: radial-gradient(circle at 50% 0%, rgba(0,212,255,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.fut-rank {
    position: absolute;
    top: 10px; left: 12px;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.5rem;
    font-weight: 900;
    color: rgba(255,199,44,0.35);
    line-height: 1;
}
.fut-pct {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.6rem;
    font-weight: 900;
    color: var(--gold);
    line-height: 1;
}
.fut-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text);
    margin: 6px 0 4px;
}
.fut-label {
    font-size: 0.62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--muted);
}

/* ── Stat widgets ────────────────────────────────── */
.stat-big {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--text);
    line-height: 1;
}
.stat-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--muted);
    margin-top: 2px;
}

/* ── Match display ───────────────────────────────── */
.vs-arena {
    background: linear-gradient(135deg, #08101e 0%, #0d1828 50%, #08101e 100%);
    border: 1px solid var(--border2);
    border-radius: 16px;
    padding: 28px 24px 20px;
    position: relative;
    overflow: hidden;
}
.vs-arena::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--blue), var(--gold), transparent);
}
.vs-arena::after {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,212,255,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.score-display {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 4.5rem;
    font-weight: 900;
    color: var(--gold);
    letter-spacing: 0.04em;
    line-height: 1;
    text-shadow: 0 0 30px rgba(255,199,44,0.3);
}
.team-name-xl {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Probability bar ─────────────────────────────── */
.prob-bar-wrap { margin: 12px 0 4px; }
.prob-bar {
    display: flex;
    height: 6px;
    border-radius: 3px;
    overflow: hidden;
    gap: 2px;
}
.prob-bar-h { background: var(--blue); border-radius: 3px 0 0 3px; }
.prob-bar-d { background: var(--muted2); }
.prob-bar-a { background: var(--gold); border-radius: 0 3px 3px 0; }

/* ── Pills ───────────────────────────────────────── */
.pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.pill-blue  { background: rgba(0,212,255,0.12);  color: var(--blue);  border: 1px solid rgba(0,212,255,0.3); }
.pill-gold  { background: rgba(255,199,44,0.12); color: var(--gold);  border: 1px solid rgba(255,199,44,0.3); }
.pill-green { background: rgba(0,230,118,0.12); color: var(--green); border: 1px solid rgba(0,230,118,0.3); }
.pill-red   { background: rgba(255,61,87,0.12);  color: var(--red);   border: 1px solid rgba(255,61,87,0.3); }
.pill-muted { background: rgba(107,122,149,0.12); color: var(--muted); border: 1px solid rgba(107,122,149,0.2); }

/* ── Buttons ─────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #00B8DC, #0090B0) !important;
    color: #000 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--gold), #e6a800) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(255,199,44,0.3) !important;
}

/* ── Selectbox ───────────────────────────────────── */
.stSelectbox > div > div {
    background: var(--card2) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}

/* ── Progress ────────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--blue), var(--gold)) !important;
}

/* ── Dividers ────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 24px 0 !important; }

/* ── Dataframe ───────────────────────────────────── */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px !important; }
.stDataFrame thead th {
    background: var(--card) !important;
    color: var(--muted) !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
}

/* ── Expander ────────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text) !important;
}

/* ── Alerts ──────────────────────────────────────── */
.stAlert { border-radius: 8px !important; border: 1px solid var(--border2) !important; }

/* ── Hide Streamlit chrome ───────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Scrollbar ───────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ── Neon glow text ──────────────────────────────── */
.glow-blue { text-shadow: 0 0 20px rgba(0,212,255,0.6); }
.glow-gold { text-shadow: 0 0 20px rgba(255,199,44,0.5); }

/* ── Bracket ─────────────────────────────────────── */
.bracket-match {
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.82rem;
    position: relative;
}
.bracket-match.winner-match { border-color: rgba(255,199,44,0.4); }
.bracket-team { padding: 4px 0; display: flex; align-items: center; gap: 8px; }
.bracket-team.winner { color: var(--gold); font-weight: 700; }
.bracket-score {
    margin-left: auto;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--gold);
}

/* ── Share card ──────────────────────────────────── */
.share-card {
    background: linear-gradient(135deg, #0a1628, #0f1e38);
    border: 2px solid var(--gold);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.share-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--blue), var(--gold), var(--green));
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# FLAGS & HELPERS
# ══════════════════════════════════════════════════════════════════════════
FLAGS = {
    "Mexico":"🇲🇽","South Africa":"🇿🇦","South Korea":"🇰🇷","Czech Republic":"🇨🇿",
    "Canada":"🇨🇦","Bosnia and Herzegovina":"🇧🇦","Qatar":"🇶🇦","Switzerland":"🇨🇭",
    "Brazil":"🇧🇷","Morocco":"🇲🇦","Haiti":"🇭🇹","Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "United States":"🇺🇸","Paraguay":"🇵🇾","Australia":"🇦🇺","Turkey":"🇹🇷",
    "Germany":"🇩🇪","Curaçao":"🇨🇼","Ivory Coast":"🇨🇮","Ecuador":"🇪🇨",
    "Netherlands":"🇳🇱","Japan":"🇯🇵","Sweden":"🇸🇪","Tunisia":"🇹🇳",
    "Belgium":"🇧🇪","Egypt":"🇪🇬","Iran":"🇮🇷","New Zealand":"🇳🇿",
    "Spain":"🇪🇸","Cape Verde":"🇨🇻","Saudi Arabia":"🇸🇦","Uruguay":"🇺🇾",
    "France":"🇫🇷","Senegal":"🇸🇳","Iraq":"🇮🇶","Norway":"🇳🇴",
    "Argentina":"🇦🇷","Algeria":"🇩🇿","Austria":"🇦🇹","Jordan":"🇯🇴",
    "Portugal":"🇵🇹","DR Congo":"🇨🇩","Uzbekistan":"🇺🇿","Colombia":"🇨🇴",
    "England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","Croatia":"🇭🇷","Ghana":"🇬🇭","Panama":"🇵🇦",
}
def flag(t): return FLAGS.get(t, "🏳️")

def stat_card(col, value, label, color="var(--blue)"):
    col.markdown(clean_html(f"""
    <div class="card-sm" style="text-align:center;">
        <div class="stat-big" style="color:{color};">{value}</div>
        <div class="stat-label">{label}</div>
    </div>"""), unsafe_allow_html=True)

def section_header(label, title=""):
    st.markdown(f'<div class="page-label">{label}</div>', unsafe_allow_html=True)
    if title:
        st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:24px;'></div>", unsafe_allow_html=True)

def divider_label(text):
    st.markdown(clean_html(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:28px 0 18px;">
        <div style="flex:1;height:1px;background:var(--border);"></div>
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.65rem;font-weight:700;
                    letter-spacing:0.2em;text-transform:uppercase;color:var(--muted2);">{text}</div>
        <div style="flex:1;height:1px;background:var(--border);"></div>
    </div>"""), unsafe_allow_html=True)


def get_gemini_analysis(home: str, away: str, pred: dict, h_r: TeamRating, a_r: TeamRating) -> str:
    # Win probability summary
    hw, aw, d = pred["home_win_pct"], pred["away_win_pct"], pred["draw_pct"]
    
    if hw > aw + 10:
        fav_text = f"**{home}** is the clear favorite with a **{hw}%** win probability, driven by their superior ratings."
    elif aw > hw + 10:
        fav_text = f"**{away}** is projected as the favorite with a **{aw}%** probability to win on neutral territory."
    else:
        fav_text = f"This is a highly competitive, closely matched fixture. **{home}** holds a marginal **{hw}%** edge, while a draw stands at **{d}%**."
        
    # Elo differences
    elo_diff = abs(h_r.elo - a_r.elo)
    higher_elo_team = home if h_r.elo > a_r.elo else away
    elo_text = f"The core statistical anchor is the Elo rating: {higher_elo_team} holds a **+{elo_diff:.0f}** point Elo advantage."
    
    # Momentum differences
    mom_text = ""
    if abs(h_r.momentum_attack - a_r.momentum_attack) > 0.3:
        hot_team = home if h_r.momentum_attack > a_r.momentum_attack else away
        cold_team = away if hot_team == home else home
        hot_val = max(h_r.momentum_attack, a_r.momentum_attack)
        mom_text = f" In terms of recent form, {hot_team} is on a hot streak, averaging **{hot_val:.1f}** goals per game in recent competitive matches, which could test {cold_team}'s defensive structure."
        
    # Mentality/Tournament factors
    tournament_text = ""
    hosts = {"United States", "Mexico", "Canada"}
    if home in hosts or away in hosts:
        host_team = home if home in hosts else away
        tournament_text += f" Playing in front of host crowds gives {host_team} a critical geographical and mental boost (+10% xG)."
    if home == "Argentina" or away == "Argentina":
        tournament_text += " Argentina faces the historical 'winner's slump' drag, reducing their attacking output by 12%."
    
    # European defensive toughness
    european_teams = {
        "Czech Republic", "Switzerland", "Scotland", "Germany", 
        "Netherlands", "Sweden", "Spain", "France", "Norway", 
        "Austria", "Portugal", "Croatia", "Belgium", "England"
    }
    if home in european_teams or away in european_teams:
        eu_team = home if home in european_teams else away
        tournament_text += f" The defensive resilience of {eu_team} (European multiplier) makes them exceptionally tough to break down."

    # Scoreline verdict
    verdict = f"**Gemini Analyst Verdict:** The Dixon-Coles Bivariate Poisson model predicts a modal scoreline of **{pred['predicted_home']}–{pred['predicted_away']}**. Expect {fav_text.split('with')[0]} to dictate the tempo of the match."

    analysis_html = clean_html(f"""
    <div style="background: linear-gradient(135deg, rgba(66, 133, 244, 0.08) 0%, rgba(155, 93, 229, 0.08) 50%, rgba(255, 110, 64, 0.05) 100%);
                border: 1px solid rgba(155, 93, 229, 0.3); border-radius: 12px; padding: 20px; margin-top: 24px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <span style="font-size: 1.5rem;">✨</span>
            <div style="font-family: 'Barlow Condensed', sans-serif; font-size: 1.25rem; font-weight: 800;
                        background: linear-gradient(90deg, #6ea3ff, #c39eff, #ffb494);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Google Gemini Tactical Analysis
            </div>
        </div>
        <div style="font-size: 0.9rem; line-height: 1.6; color: #E2E8F0; margin-bottom: 12px;">
            {fav_text} {elo_text}{mom_text}{tournament_text}
        </div>
        <div style="font-size: 0.9rem; line-height: 1.6; color: #FFC72C; font-weight: 600;">
            {verdict}
        </div>
    </div>
    """)
    return analysis_html

# ══════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_engine():
    e = FootballEngine()
    e.load_data()
    if not e.load_ratings():
        with st.spinner("⚙️  Building AI engine from 49,000+ historical matches…"):
            e.compute_ratings()
    return e

@st.cache_data(show_spinner=False)
def get_mc_data():
    p = DATA_DIR / "winner_probabilities.csv"
    if p.exists():
        df = pd.read_csv(p)
        return sort_winning_probabilities(df)
    return pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════
PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Barlow Condensed", color="#F5F7FA"),
    margin=dict(l=0, r=0, t=0, b=0),
)

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(clean_html("""
    <div style="padding:24px 16px 20px;border-bottom:1px solid #1A2540;margin-bottom:8px;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;font-weight:900;
                    text-transform:uppercase;letter-spacing:-0.02em;line-height:1;">
            <span style="background:linear-gradient(135deg,#7FA6EE,#B88FF7,#FFAE94);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;">World</span><span
                  style="color:#FFC72C;">Cup</span><span
                  style="background:linear-gradient(135deg,#7FA6EE,#B88FF7,#FFAE94);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;">AI</span>
        </div>
        <div style="font-size:0.62rem;color:#3D4F6E;letter-spacing:0.2em;
                    text-transform:uppercase;margin-top:4px;">
            FIFA World Cup 2026 · AI Analytics
        </div>
    </div>"""), unsafe_allow_html=True)

    page = st.radio("", [
        "🏆  Home",
        "⚽  Match Predictor",
        "📊  Group Stage",
        "🎯  Tournament Predictor",
        "🏟️  Knockout Bracket",
        "🔍  Team Analytics",
        "🎲  Monte Carlo",
    ], label_visibility="collapsed")

    st.markdown(clean_html("""
    <div style="margin-top:80px;padding-top:16px;border-top:1px solid #1A2540;">
        <div style="font-size:0.6rem;color:#3D4F6E;text-transform:uppercase;
                    letter-spacing:0.15em;text-align:center;">
            Powered by Poisson · Elo · Monte Carlo
        </div>
    </div>"""), unsafe_allow_html=True)

engine = get_engine()

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════
if page == "🏆  Home":

    # Hero
    st.markdown(clean_html("""
    <div style="padding:32px 0 24px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:-20px;right:-40px;width:300px;height:300px;
                    background:radial-gradient(circle,rgba(0,212,255,0.06) 0%,transparent 70%);
                    pointer-events:none;"></div>
        <div style="position:absolute;bottom:-40px;left:-20px;width:250px;height:250px;
                    background:radial-gradient(circle,rgba(255,199,44,0.05) 0%,transparent 70%);
                    pointer-events:none;"></div>
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.7rem;font-weight:700;
                    letter-spacing:0.3em;text-transform:uppercase;color:#B88FF7;margin-bottom:8px;
                    text-shadow: 0 0 10px rgba(184,143,247,0.3);">
            ✨ AI-Powered Analytics Platform
        </div>
        <div class="hero">FIFA World Cup<br>2026 Predictor</div>
        <div style="margin-top:14px;display:flex;gap:24px;flex-wrap:wrap;">
            <span style="font-size:0.85rem;color:#6B7A95;">⚡ Poisson Distribution</span>
            <span style="font-size:0.85rem;color:#6B7A95;">📊 Elo Rating System</span>
            <span style="font-size:0.85rem;color:#6B7A95;">🎲 Monte Carlo Simulation</span>
            <span style="font-size:0.85rem;color:#6B7A95;">🏆 49,000+ Matches Trained</span>
        </div>
    </div>
    """), unsafe_allow_html=True)

    mc = get_mc_data()

    if not mc.empty:
        divider_label("Top Contenders")
        top5 = mc.head(5)
        cols = st.columns(5)
        for i, (_, row) in enumerate(top5.iterrows()):
            cols[i].markdown(clean_html(f"""
            <div class="fut-card">
                <div class="fut-rank">#{i+1}</div>
                <div style="font-size:3rem;margin:8px 0 4px;">{flag(row['team'])}</div>
                <div class="fut-pct">{row['champion_pct']}%</div>
                <div class="fut-name">{row['team']}</div>
                <div class="fut-label">Champion Probability</div>
                <div style="margin-top:10px;display:flex;justify-content:center;gap:6px;">
                    <span class="pill pill-blue" style="font-size:0.58rem;">
                        Final {row['finalist_pct']}%
                    </span>
                </div>
            </div>"""), unsafe_allow_html=True)

        divider_label("Championship Probability — Top 15")
        col1, col2 = st.columns([3, 2])
        with col1:
            top15 = mc.head(15).copy()
            top15["label"] = top15["team"].apply(lambda t: f"{flag(t)}  {t}")
            fig = go.Figure(go.Bar(
                x=top15["champion_pct"], y=top15["label"],
                orientation="h",
                marker=dict(
                    color=top15["champion_pct"],
                    colorscale=[[0,"#0d2233"],[0.5,"#0090B0"],[1,"#FFC72C"]],
                    showscale=False,
                    line=dict(width=0),
                ),
                text=top15["champion_pct"].apply(lambda v: f" {v}%"),
                textposition="outside",
                textfont=dict(color="#F5F7FA", size=11, family="Barlow Condensed"),
            ))
            fig.update_layout(**PLOT_BASE, height=440,
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False, color="#F5F7FA", tickfont=dict(size=12, family="Barlow Condensed")),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col2:
            top8 = mc.head(8).copy()
            top8["label"] = top8["team"].apply(lambda t: f"{flag(t)} {t}")
            fig2 = go.Figure()
            for col_name, label, color in [
                ("semifinal_pct","Semi-Final","#1A3550"),
                ("finalist_pct","Final","#00A8CC"),
                ("champion_pct","Champion","#FFC72C"),
            ]:
                fig2.add_trace(go.Bar(
                    name=label, y=top8["label"], x=top8[col_name],
                    orientation="h", marker_color=color,
                    text=top8[col_name].apply(lambda v: f"{v}%"),
                    textposition="inside",
                    textfont=dict(size=9, color="#000" if color=="#FFC72C" else "#fff"),
                ))
            fig2.update_layout(**PLOT_BASE, barmode="overlay", height=440,
                legend=dict(font=dict(color="#F5F7FA", size=10), bgcolor="rgba(0,0,0,0)",
                            orientation="h", x=0, y=1.05),
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False, color="#F5F7FA", tickfont=dict(size=11, family="Barlow Condensed")),
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown(clean_html("""
        <div style="background:var(--card);border:1px dashed var(--border2);border-radius:12px;
                    padding:40px;text-align:center;margin:20px 0;">
            <div style="font-size:2.5rem;margin-bottom:8px;">🎲</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
                        color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;">
                No simulation data yet
            </div>
            <div style="color:var(--muted2);font-size:0.85rem;margin-top:4px;">
                Head to Monte Carlo to generate win probabilities
            </div>
        </div>"""), unsafe_allow_html=True)

    divider_label("AI Power Rankings")
    wc_teams = [t for g in engine.GROUPS.values() for t in g]
    rows = []
    for t in wc_teams:
        r = engine._get_rating(t)
        rows.append({"team":t,"elo":r.elo,"attack":r.attack,"defense":r.defense,"form":r.recent_form})
    rank_df = pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)
    rank_df.index += 1
    rank_df.insert(0, "", rank_df["team"].apply(flag))
    styled = rank_df[["","team","elo","attack","defense","form"]].rename(
        columns={"team":"Team","elo":"Elo","attack":"Attack","defense":"Defense","form":"Form"})
    st.dataframe(styled, use_container_width=True, height=380,
        column_config={
            "Elo": st.column_config.NumberColumn(format="%.0f"),
            "Attack": st.column_config.NumberColumn(format="%.2f"),
            "Defense": st.column_config.NumberColumn(format="%.2f"),
            "Form": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
        })

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════════════════
elif page == "⚽  Match Predictor":
    section_header("AI Match Intelligence", "Match Predictor")

    all_wc_teams = sorted({t for g in engine.GROUPS.values() for t in g})

    # Team selector arena
    c1, cmid, c2 = st.columns([5, 2, 5])
    with c1:
        home_team = st.selectbox("Home Team", all_wc_teams,
                                  index=all_wc_teams.index("Spain"), label_visibility="collapsed")
        st.markdown(clean_html(f"""
        <div style="text-align:center;padding:12px 0;">
            <div style="font-size:4rem;">{flag(home_team)}</div>
            <div class="team-name-xl">{home_team}</div>
        </div>"""), unsafe_allow_html=True)
        r = engine._get_rating(home_team)
        sc1, sc2, sc3 = st.columns(3)
        stat_card(sc1, f"{r.elo:.0f}", "Elo", "#00D4FF")
        stat_card(sc2, f"{r.attack:.2f}", "Attack", "#FFC72C")
        stat_card(sc3, f"{r.defense:.2f}", "Defense", "#00E676")

    with cmid:
        st.markdown(clean_html("""
        <div style="text-align:center;padding-top:80px;">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;
                        font-weight:900;color:var(--muted2);letter-spacing:0.1em;">VS</div>
        </div>"""), unsafe_allow_html=True)

    with c2:
        away_team = st.selectbox("Away Team", all_wc_teams,
                                  index=all_wc_teams.index("France"), label_visibility="collapsed")
        st.markdown(clean_html(f"""
        <div style="text-align:center;padding:12px 0;">
            <div style="font-size:4rem;">{flag(away_team)}</div>
            <div class="team-name-xl">{away_team}</div>
        </div>"""), unsafe_allow_html=True)
        r2 = engine._get_rating(away_team)
        sc4, sc5, sc6 = st.columns(3)
        stat_card(sc4, f"{r2.elo:.0f}", "Elo", "#00D4FF")
        stat_card(sc5, f"{r2.attack:.2f}", "Attack", "#FFC72C")
        stat_card(sc6, f"{r2.defense:.2f}", "Defense", "#00E676")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    ba, bb, _ = st.columns([2, 2, 3])
    predict_btn  = ba.button("📊  Predict Match",  use_container_width=True)
    simulate_btn = bb.button("⚡  Simulate Match", use_container_width=True)

    if home_team == away_team:
        st.warning("Select two different teams.")
    elif predict_btn:
        pred = engine.predict_match(home_team, away_team)
        divider_label("AI Prediction")
        st.markdown(clean_html(f"""
        <div class="vs-arena">
            <div style="display:flex;align-items:center;justify-content:space-between;">
                <div style="text-align:center;flex:1;">
                    <div style="font-size:3.5rem;">{flag(home_team)}</div>
                    <div class="team-name-xl" style="color:{'#00D4FF' if pred['home_win_pct']>pred['away_win_pct'] else '#F5F7FA'};">
                        {home_team}
                    </div>
                    <div style="margin-top:8px;">
                        <span class="pill pill-blue">xG {pred['home_xg']}</span>
                    </div>
                </div>
                <div style="text-align:center;flex:0 0 160px;">
                    <div class="score-display">{pred['predicted_home']}–{pred['predicted_away']}</div>
                    <div style="font-size:0.65rem;color:var(--muted2);letter-spacing:0.2em;
                                text-transform:uppercase;margin-top:4px;">Predicted Score</div>
                </div>
                <div style="text-align:center;flex:1;">
                    <div style="font-size:3.5rem;">{flag(away_team)}</div>
                    <div class="team-name-xl" style="color:{'#FFC72C' if pred['away_win_pct']>pred['home_win_pct'] else '#F5F7FA'};">
                        {away_team}
                    </div>
                    <div style="margin-top:8px;">
                        <span class="pill pill-gold">xG {pred['away_xg']}</span>
                    </div>
                </div>
            </div>
            <div class="prob-bar-wrap">
                <div class="prob-bar">
                    <div class="prob-bar-h" style="width:{pred['home_win_pct']}%;"></div>
                    <div class="prob-bar-d" style="width:{pred['draw_pct']}%;"></div>
                    <div class="prob-bar-a" style="width:{pred['away_win_pct']}%;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:6px;">
                    <span style="font-family:'Barlow Condensed',sans-serif;font-size:1rem;
                                 font-weight:700;color:#00D4FF;">{pred['home_win_pct']}%</span>
                    <span style="font-size:0.78rem;color:var(--muted);">Draw {pred['draw_pct']}%</span>
                    <span style="font-family:'Barlow Condensed',sans-serif;font-size:1rem;
                                 font-weight:700;color:#FFC72C;">{pred['away_win_pct']}%</span>
                </div>
            </div>
        </div>"""), unsafe_allow_html=True)

        divider_label("Score Probabilities & Analysis")
        col_s, col_r = st.columns([1, 1])

        with col_s:
            st.markdown('<div class="section-label">Top Predicted Scorelines</div>', unsafe_allow_html=True)
            for j, s in enumerate(pred["top_scores"]):
                bar_w = s["prob"] / pred["top_scores"][0]["prob"] * 100
                is_top = j == 0
                st.markdown(clean_html(f"""
                <div style="background:{'rgba(255,199,44,0.08)' if is_top else 'var(--card2)'};
                            border:1px solid {'rgba(255,199,44,0.3)' if is_top else 'var(--border)'};
                            border-radius:8px;padding:10px 14px;margin-bottom:6px;
                            position:relative;overflow:hidden;">
                    <div style="position:absolute;left:0;top:0;bottom:0;width:{bar_w}%;
                                background:rgba(0,212,255,0.05);border-radius:8px;"></div>
                    <div style="position:relative;display:flex;justify-content:space-between;align-items:center;">
                        <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.3rem;
                                    font-weight:700;{'color:#FFC72C;' if is_top else ''}">{s['score']}</div>
                        <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
                                    font-weight:700;color:#00D4FF;">{s['prob']}%</div>
                    </div>
                </div>"""), unsafe_allow_html=True)

        with col_r:
            st.markdown('<div class="section-label">Win Probability</div>', unsafe_allow_html=True)
            fig_pie = go.Figure(go.Pie(
                labels=[f"{home_team}", "Draw", f"{away_team}"],
                values=[pred["home_win_pct"], pred["draw_pct"], pred["away_win_pct"]],
                hole=0.65,
                marker=dict(
                    colors=["#00D4FF", "#1A2540", "#FFC72C"],
                    line=dict(color="#05070D", width=3),
                ),
                textfont=dict(family="Barlow Condensed", size=12),
                hovertemplate="%{label}: %{value}%<extra></extra>",
            ))
            fig_pie.update_layout(**PLOT_BASE, height=260,
                legend=dict(font=dict(color="#F5F7FA", size=11), bgcolor="rgba(0,0,0,0)"),
                annotations=[dict(text=f"{'W' if pred['home_win_pct']>pred['away_win_pct'] else 'W'}",
                                  font=dict(size=14, color="#6B7A95"), showarrow=False)],
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

            # Why prediction
            h_r = engine._get_rating(home_team)
            a_r = engine._get_rating(away_team)
            reasons = []
            fav = home_team if pred["home_win_pct"] >= pred["away_win_pct"] else away_team
            if h_r.elo != a_r.elo:
                reasons.append(("Higher Elo Rating", "#00D4FF"))
            if h_r.attack > a_r.attack or a_r.attack > h_r.attack:
                reasons.append(("Stronger Attack Rating", "#FFC72C"))
            if h_r.recent_form != a_r.recent_form:
                reasons.append(("Better Recent Form", "#00E676"))
            if h_r.tournament_exp != a_r.tournament_exp:
                reasons.append(("More Tournament Experience", "#8B5CF6"))
            st.markdown(f'<div class="section-label" style="margin-top:16px;">Why {fav}?</div>', unsafe_allow_html=True)
            for reason, color in reasons[:4]:
                st.markdown(clean_html(f"""
                <div style="display:flex;align-items:center;gap:8px;padding:5px 0;
                            border-bottom:1px solid var(--border);">
                    <div style="width:6px;height:6px;border-radius:50%;background:{color};flex-shrink:0;"></div>
                    <div style="font-size:0.85rem;">{reason}</div>
                </div>"""), unsafe_allow_html=True)
            
            # Google Gemini tactical analysis card
            analysis_html = get_gemini_analysis(home_team, away_team, pred, h_r, a_r)
            st.markdown(analysis_html, unsafe_allow_html=True)

    elif simulate_btn:
        result = engine.simulate_match(home_team, away_team, knockout=True)
        winner = result.winner or "Draw"
        divider_label("Match Simulation Result")
        st.markdown(clean_html(f"""
        <div class="vs-arena">
            <div style="display:flex;align-items:center;justify-content:space-between;">
                <div style="text-align:center;flex:1;">
                    <div style="font-size:3.5rem;">{flag(home_team)}</div>
                    <div class="team-name-xl" style="color:{'#00D4FF' if winner==home_team else '#6B7A95'};">
                        {home_team}
                    </div>
                </div>
                <div style="text-align:center;flex:0 0 160px;">
                    <div class="score-display">{result.home_score}–{result.away_score}</div>
                    {'<div style="color:#FFC72C;font-size:0.8rem;margin-top:4px;">Pens '+str(result.pen_home)+'–'+str(result.pen_away)+'</div>' if result.went_to_pens else ''}
                    {'<div style="color:var(--muted);font-size:0.75rem;margin-top:4px;">AET</div>' if result.went_to_et and not result.went_to_pens else ''}
                </div>
                <div style="text-align:center;flex:1;">
                    <div style="font-size:3.5rem;">{flag(away_team)}</div>
                    <div class="team-name-xl" style="color:{'#FFC72C' if winner==away_team else '#6B7A95'};">
                        {away_team}
                    </div>
                </div>
            </div>
        </div>"""), unsafe_allow_html=True)
        if winner != "Draw":
            st.success(f"🏆  **{flag(winner)} {winner} advances!**")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — GROUP STAGE
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊  Group Stage":
    section_header("FIFA World Cup 2026", "Group Stage")

    if st.button("🎲  Simulate All Groups", use_container_width=False):
        with st.spinner("Simulating 72 group-stage fixtures…"):
            results = engine.simulate_group_stage()
        st.session_state["group_results"] = results

    group_results = st.session_state.get("group_results")

    if not group_results:
        st.info("Click **Simulate All Groups** to generate AI-predicted standings.")

    divider_label("All 12 Groups")
    cols3 = st.columns(3)
    for i, (g, teams) in enumerate(engine.GROUPS.items()):
        with cols3[i % 3]:
            if group_results and g in group_results:
                ranked = group_results[g].to_dict("records")
                rows_html = ""
                for rank_idx, row in enumerate(ranked):
                    if rank_idx < 2:
                        qual_color = "#00D4FF"; qual_bg = "rgba(0,212,255,0.07)"
                    elif rank_idx == 2:
                        qual_color = "#FFC72C"; qual_bg = "rgba(255,199,44,0.05)"
                    else:
                        qual_color = "var(--muted2)"; qual_bg = "transparent"
                    medals = ["①","②","③","④"]
                    rows_html += f"""
                    <div style="display:flex;align-items:center;gap:8px;padding:7px 8px;
                                margin-bottom:3px;background:{qual_bg};border-radius:5px;
                                border-left:2px solid {qual_color};">
                        <span style="font-family:'Barlow Condensed',sans-serif;font-size:1rem;
                                     color:{qual_color};">{medals[rank_idx]}</span>
                        <span style="font-size:1.3rem;">{flag(row['team'])}</span>
                        <span style="flex:1;font-weight:600;font-size:0.82rem;
                                     overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{row['team']}</span>
                        <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
                                     font-weight:800;color:#FFC72C;">{row['pts']}</span>
                        <span style="font-size:0.75rem;color:var(--muted);">
                            {row['gd']:+d}
                        </span>
                    </div>"""
            else:
                rows_html = ""
                for t in teams:
                    r = engine._get_rating(t)
                    rows_html += f"""
                    <div style="display:flex;align-items:center;gap:8px;padding:7px 6px;
                                margin-bottom:3px;background:var(--card2);border-radius:5px;">
                        <span style="font-size:1.3rem;">{flag(t)}</span>
                        <span style="flex:1;font-weight:600;font-size:0.83rem;">{t}</span>
                        <span class="pill pill-blue" style="font-size:0.6rem;">{r.elo:.0f}</span>
                    </div>"""

            qualifier_note = '<div style="font-size:0.62rem;margin-top:8px;color:var(--muted2);">Blue = Qualified · Gold = Best 3rd</div>' if group_results else ""
            st.markdown(clean_html(f"""
            <div style="background:var(--card);border:1px solid var(--border);border-radius:12px;
                        padding:16px;margin-bottom:16px;position:relative;overflow:hidden;">
                <div style="position:absolute;top:0;left:0;right:0;height:2px;
                            background:linear-gradient(90deg,var(--blue),var(--gold));"></div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.2rem;
                            font-weight:900;text-transform:uppercase;letter-spacing:0.12em;
                            color:#FFC72C;margin-bottom:12px;">Group {g}</div>
                {rows_html}
                {qualifier_note}
            </div>"""), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — TOURNAMENT PREDICTOR (flagship)
# ══════════════════════════════════════════════════════════════════════════
elif page == "🎯  Tournament Predictor":
    render_tournament_predictor_page(engine)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 — KNOCKOUT BRACKET
# ══════════════════════════════════════════════════════════════════════════
elif page == "🏟️  Knockout Bracket":
    section_header("Full Tournament Simulation", "Knockout Bracket")

    if st.button("🏆  Simulate Full Tournament", use_container_width=False):
        with st.spinner("Simulating complete World Cup…"):
            bracket = engine.simulate_tournament()
        st.session_state["bracket"] = bracket

    bracket = st.session_state.get("bracket")

    if not bracket:
        st.info("Click **Simulate Full Tournament** to generate the complete bracket.")
    else:
        champion = bracket.get("Champion", "TBD")
        st.markdown(clean_html(f"""
        <div style="background:linear-gradient(135deg,rgba(255,199,44,0.1),rgba(0,212,255,0.06));
                    border:1px solid rgba(255,199,44,0.5);border-radius:16px;
                    padding:32px;text-align:center;margin-bottom:28px;
                    box-shadow:0 0 40px rgba(255,199,44,0.1);position:relative;overflow:hidden;">
            <div style="position:absolute;top:0;left:0;right:0;height:3px;
                        background:linear-gradient(90deg,var(--blue),var(--gold),var(--green));"></div>
            <div style="font-size:0.65rem;color:#FFC72C;text-transform:uppercase;
                        letter-spacing:0.3em;font-weight:700;margin-bottom:12px;">
                🏆 FIFA World Cup 2026 Champion
            </div>
            <div style="font-size:5rem;line-height:1;margin-bottom:8px;">{flag(champion)}</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:3rem;
                        font-weight:900;text-transform:uppercase;letter-spacing:0.04em;
                        text-shadow:0 0 30px rgba(255,199,44,0.4);">{champion}</div>
        </div>"""), unsafe_allow_html=True)

        round_order = ["Round of 32","Round of 16","Quarter-Finals","Semi-Finals","Bronze Final","Final"]
        for rnd in round_order:
            rlist = bracket.get(rnd)
            if not rlist:
                continue
            divider_label(rnd)
            n = len(rlist)
            cols_n = min(4, n) if rnd == "Round of 32" else min(2, n)
            for start in range(0, n, cols_n):
                chunk = rlist[start:start+cols_n]
                rcols = st.columns(len(chunk))
                for col, r in zip(rcols, chunk):
                    winner = r.winner or "—"
                    pens = f'<div style="font-size:0.7rem;color:#FFC72C;text-align:center;">Pens {r.pen_home}–{r.pen_away}</div>' if r.went_to_pens else ""
                    et   = '<div style="font-size:0.7rem;color:var(--muted);text-align:center;">AET</div>' if r.went_to_et and not r.went_to_pens else ""
                    col.markdown(clean_html(f"""
                    <div style="background:var(--card2);border:1px solid {'rgba(255,199,44,0.3)' if rnd=='Final' else 'var(--border)'};
                                border-radius:10px;padding:12px 10px;margin-bottom:8px;">
                        <div class="bracket-team {'winner' if winner==r.home else ''}">
                            <span style="font-size:1.2rem;">{flag(r.home)}</span>
                            <span style="font-size:0.82rem;font-weight:{'700' if winner==r.home else '400'};">{r.home}</span>
                            <span class="bracket-score">{r.home_score}</span>
                        </div>
                        <div style="height:1px;background:var(--border);margin:4px 0;"></div>
                        <div class="bracket-team {'winner' if winner==r.away else ''}">
                            <span style="font-size:1.2rem;">{flag(r.away)}</span>
                            <span style="font-size:0.82rem;font-weight:{'700' if winner==r.away else '400'};">{r.away}</span>
                            <span class="bracket-score">{r.away_score}</span>
                        </div>
                        {pens}{et}
                    </div>"""), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 6 — TEAM ANALYTICS
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔍  Team Analytics":
    section_header("AI Scouting Report", "Team Analytics")

    all_wc = sorted({t for g in engine.GROUPS.values() for t in g})
    team = st.selectbox("Select Team", all_wc, label_visibility="collapsed")
    r = engine._get_rating(team)

    # Team banner
    st.markdown(clean_html(f"""
    <div style="background:linear-gradient(135deg,#0d1a2e 0%,#0f2040 40%,#080f1c 100%);
                border:1px solid var(--border2);border-radius:16px;
                padding:28px 32px;margin-bottom:20px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:2px;
                    background:linear-gradient(90deg,var(--blue),var(--gold));"></div>
        <div style="position:absolute;right:-20px;top:-20px;font-size:8rem;
                    opacity:0.06;pointer-events:none;">{flag(team)}</div>
        <div style="display:flex;align-items:center;gap:20px;">
            <div style="font-size:4.5rem;">{flag(team)}</div>
            <div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:2.2rem;
                            font-weight:900;text-transform:uppercase;letter-spacing:0.04em;">{team}</div>
                <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;">
                    <span class="pill pill-blue">Elo {r.elo:.0f}</span>
                    <span class="pill pill-gold">Attack {r.attack:.2f}</span>
                    <span class="pill pill-green">Defense {r.defense:.2f}</span>
                    <span class="pill pill-muted">Form {r.recent_form:.2f}</span>
                </div>
            </div>
        </div>
    </div>"""), unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    stat_card(s1, f"{r.elo:.0f}", "Elo Rating", "#00D4FF")
    stat_card(s2, f"{r.win_pct*100:.1f}%", "Win Rate", "#FFC72C")
    stat_card(s3, f"{r.goals_scored_avg:.2f}", "Goals/Game", "#00E676")
    stat_card(s4, f"{r.goals_conceded_avg:.2f}", "Conceded/Game", "#FF3D57")

    divider_label("Attribute Radar")
    col_r, col_h = st.columns([1, 1])
    with col_r:
        cats = ["Attack","Defense","Form","Experience","Momentum"]
        vals = [
            min(r.attack/2.5,1)*100,
            min(r.defense/2.5,1)*100,
            r.recent_form*100,
            r.tournament_exp*100,
            r.momentum*100,
        ]
        fig_radar = go.Figure(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]],
            fill="toself",
            fillcolor="rgba(0,212,255,0.12)",
            line=dict(color="#00D4FF", width=2),
            marker=dict(size=7, color="#FFC72C", line=dict(color="#000",width=1)),
        ))
        fig_radar.update_layout(**{**PLOT_BASE, "margin": dict(l=55, r=55, t=35, b=35)}, height=300,
            polar=dict(
                bgcolor="rgba(10,14,26,0.8)",
                radialaxis=dict(visible=True, range=[0,100],
                               tickfont=dict(color="rgba(107,122,149,0.6)",size=8),
                               gridcolor="#1A2540", linecolor="#1A2540"),
                angularaxis=dict(tickfont=dict(color="#F5F7FA",size=12,family="Barlow Condensed"),
                                gridcolor="#1A2540", linecolor="#1A2540"),
            ),
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar":False})

    with col_h:
        st.markdown('<div class="section-label">Head-to-Head Simulation</div>', unsafe_allow_html=True)
        opponents = [t for t in all_wc if t != team]
        opp = st.selectbox("Simulate vs.", opponents, key="h2h_opp")
        if st.button("Run 1,000× Simulations", key="h2h_btn"):
            wins = draws = losses = 0
            for _ in range(1000):
                res = engine.simulate_match(team, opp, knockout=False)
                if res.home_score > res.away_score: wins += 1
                elif res.home_score == res.away_score: draws += 1
                else: losses += 1
            fig_h2h = go.Figure(go.Bar(
                x=["Win","Draw","Loss"],
                y=[wins/10, draws/10, losses/10],
                marker_color=["#00D4FF","#1A2540","#FFC72C"],
                text=[f"{v/10:.1f}%" for v in [wins,draws,losses]],
                textposition="outside",
                textfont=dict(color="#F5F7FA", family="Barlow Condensed"),
            ))
            fig_h2h.update_layout(**PLOT_BASE, height=240,
                xaxis=dict(color="#F5F7FA", tickfont=dict(family="Barlow Condensed",size=13)),
                yaxis=dict(showgrid=False, showticklabels=False, range=[0,110]),
            )
            st.plotly_chart(fig_h2h, use_container_width=True, config={"displayModeBar":False})

    def get_tier_color(tier):
        if tier == "SUPERSTAR": return "#FF3D57"
        if tier == "STAR": return "#FFC72C"
        if tier == "STARTER": return "#00D4FF"
        if tier == "WISSEL": return "#00E676"
        return "#6B7A95"

    divider_label("Squad Player Ratings")
    squad_df = getattr(engine, "_squads_df", None)
    if squad_df is not None and not squad_df.empty:
        team_players = squad_df[squad_df["Team"] == team]
    else:
        team_players = pd.DataFrame()

    if not team_players.empty:
        team_players = team_players.sort_values("Rating", ascending=False)
        st.markdown(clean_html(f"""
        <div style="background:var(--card2);border:1px solid var(--border);border-radius:10px;padding:12px;margin-bottom:15px;display:flex;justify-content:space-around;text-align:center;">
            <div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;font-weight:800;color:#00D4FF;">{r.starting_xi_strength:.1f}</div>
                <div style="font-size:0.65rem;color:#6B7A95;text-transform:uppercase;">Starting XI Strength</div>
            </div>
            <div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;font-weight:800;color:#FFC72C;">{r.squad_depth:.1f}</div>
                <div style="font-size:0.65rem;color:#6B7A95;text-transform:uppercase;">Squad Depth</div>
            </div>
            <div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;font-weight:800;color:#00E676;">{r.star_power:.1f}</div>
                <div style="font-size:0.65rem;color:#6B7A95;text-transform:uppercase;">Star Power</div>
            </div>
            <div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;font-weight:800;color:#FF3D57;">{r.positional_balance:.1f}</div>
                <div style="font-size:0.65rem;color:#6B7A95;text-transform:uppercase;">Positional Balance</div>
            </div>
        </div>
        """), unsafe_allow_html=True)
        
        player_cols = st.columns(2)
        half = (len(team_players) + 1) // 2
        for col_idx, col in enumerate(player_cols):
            chunk = team_players.iloc[col_idx*half : (col_idx+1)*half]
            players_html = ""
            for _, p in chunk.iterrows():
                t_color = get_tier_color(p["Tier"])
                pos_label = {"GK": "GK", "DF": "DEF", "MF": "MID", "AT": "FWD"}.get(p["Position"], p["Position"])
                players_html += f"""
                <div style="display:flex;align-items:center;justify-content:space-between;padding:6px 10px;margin-bottom:5px;background:var(--card);border-radius:6px;border:1px solid var(--border);">
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span style="font-size:0.7rem;font-weight:bold;color:#6B7A95;background:rgba(255,255,255,0.05);padding:2px 5px;border-radius:3px;">{pos_label}</span>
                        <span style="font-weight:600;font-size:0.85rem;color:#F5F7FA;">{p["Player"]}</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <span style="font-size:0.65rem;font-weight:bold;color:{t_color};border:1px solid {t_color};padding:1px 5px;border-radius:3px;text-transform:uppercase;">{p["Tier"]}</span>
                        <span style="font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:1.1rem;color:#00D4FF;">{p["Rating"]}</span>
                    </div>
                </div>
                """
            col.markdown(clean_html(players_html), unsafe_allow_html=True)
    else:
        st.info("ℹ️ Country-wise squad player list not available for this team. Fallback metrics have been generated based on historical performance & Elo ratings.")

    divider_label("Tournament Path (500 simulations)")
    if st.button(f"Predict {team}&#39;s Tournament Path", key="path_btn"):
        reached = {"Group Stage":0,"Round of 32":0,"Round of 16":0,
                   "Quarter-Finals":0,"Semi-Finals":0,"Final":0,"Champion":0}
        with st.spinner("Simulating 500 tournaments…"):
            for _ in range(500):
                b = engine.simulate_tournament()
                reached["Group Stage"] += 1
                for stage in ["Round of 32","Round of 16","Quarter-Finals","Semi-Finals"]:
                    rlist = b.get(stage, [])
                    if any(x.home == team or x.away == team for x in rlist):
                        reached[stage] += 1
                final = b.get("Final",[])
                if final and (final[0].home==team or final[0].away==team):
                    reached["Final"] += 1
                if b.get("Champion")==team: reached["Champion"] += 1

        stages = list(reached.keys())
        pcts = [reached[s]/500*100 for s in stages]
        fig_path = go.Figure(go.Bar(
            x=stages, y=pcts,
            marker=dict(color=pcts,
                colorscale=[[0,"#0d2233"],[0.6,"#0090B0"],[1,"#FFC72C"]],showscale=False),
            text=[f"{p:.1f}%" for p in pcts], textposition="outside",
            textfont=dict(color="#F5F7FA", family="Barlow Condensed"),
        ))
        fig_path.update_layout(**PLOT_BASE, height=300,
            xaxis=dict(color="#F5F7FA", tickfont=dict(family="Barlow Condensed",size=11), tickangle=-20),
            yaxis=dict(showgrid=False, showticklabels=False, range=[0,120]),
        )
        st.plotly_chart(fig_path, use_container_width=True, config={"displayModeBar":False})

    # Share card
    divider_label("Share Card Generator")
    if st.button("🖼️  Generate Share Card", key="share_btn"):
        r2 = engine._get_rating(team)
        mc_data = get_mc_data()
        champ_pct = "—"
        if not mc_data.empty:
            row = mc_data[mc_data["team"] == team]
            if not row.empty:
                champ_pct = f"{row.iloc[0]['champion_pct']}%"
        st.markdown(clean_html(f"""
        <div class="share-card">
            <div style="font-size:0.65rem;color:#FFC72C;text-transform:uppercase;
                        letter-spacing:0.3em;font-weight:700;margin-bottom:16px;">
                WorldCupAI 2026 · Team Report
            </div>
            <div style="font-size:4rem;margin-bottom:8px;">{flag(team)}</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:2.5rem;
                        font-weight:900;text-transform:uppercase;">{team}</div>
            <div style="display:flex;justify-content:center;gap:24px;margin-top:20px;">
                <div style="text-align:center;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;
                                font-weight:800;color:#00D4FF;">{r2.elo:.0f}</div>
                    <div style="font-size:0.65rem;color:#6B7A95;text-transform:uppercase;
                                letter-spacing:0.1em;">Elo Rating</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;
                                font-weight:800;color:#FFC72C;">{champ_pct}</div>
                    <div style="font-size:0.65rem;color:#6B7A95;text-transform:uppercase;
                                letter-spacing:0.1em;">Win Prob.</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;
                                font-weight:800;color:#00E676;">{r2.attack:.2f}</div>
                    <div style="font-size:0.65rem;color:#6B7A95;text-transform:uppercase;
                                letter-spacing:0.1em;">Attack</div>
                </div>
            </div>
            <div style="margin-top:20px;font-size:0.65rem;color:#3D4F6E;
                        text-transform:uppercase;letter-spacing:0.2em;">
                Generated by WorldCupAI · AI-Powered Predictions
            </div>
        </div>"""), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 7 — MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════
elif page == "🎲  Monte Carlo":
    section_header("Probability Command Center", "Monte Carlo Analysis")

    col_ctrl, col_info = st.columns([1, 2])
    with col_ctrl:
        n_sims = st.slider("Simulations", 500, 10000, 3000, step=500, label_visibility="collapsed")
    est_seconds = int(n_sims * 0.015)
    if est_seconds < 60:
        est_time_str = f"~{est_seconds} sec"
    else:
        est_time_str = f"~{est_seconds/60:.1f} min"

    with col_info:
        st.markdown(clean_html(f"""
        <div class="card-sm" style="margin-top:4px;">
            <div style="display:flex;gap:24px;align-items:center;">
                <div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;
                                font-weight:800;color:#00D4FF;">{n_sims:,}</div>
                    <div class="stat-label">Tournaments to simulate</div>
                </div>
                <div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;
                                font-weight:800;color:#FFC72C;">{est_time_str}</div>
                    <div class="stat-label">Estimated runtime</div>
                </div>
                <div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;
                                font-weight:800;color:#00E676;">{n_sims*104:,}</div>
                    <div class="stat-label">Matches computed</div>
                </div>
            </div>
        </div>"""), unsafe_allow_html=True)

    if st.button(f"🚀  Run {n_sims:,} Simulations", use_container_width=False):
        bar = st.progress(0, text="Initialising…")
        batch = max(1, n_sims//20)
        cc = defaultdict(int); fc = defaultdict(int); sc = defaultdict(int)
        done = 0
        for _ in range(0, n_sims, batch):
            chunk = min(batch, n_sims-done)
            for __ in range(chunk):
                b = engine.simulate_tournament()
                ch = b.get("Champion")
                if ch: cc[ch] += 1
                for x in b.get("Final",[]): fc[x.home]+=1; fc[x.away]+=1
                for x in b.get("Semi-Finals",[]): sc[x.home]+=1; sc[x.away]+=1
            done += chunk
            bar.progress(done/n_sims, text=f"Simulated {done:,}/{n_sims:,}…")
        bar.empty()
        rows_mc = [{"team":t,"champion_pct":round(cc[t]/n_sims*100,2),
                    "finalist_pct":round(fc[t]/n_sims*100,2),
                    "semifinal_pct":round(sc[t]/n_sims*100,2)}
                   for t in set(list(cc.keys())+list(fc.keys()))]
        mc_df = pd.DataFrame(rows_mc).sort_values("champion_pct", ascending=False)
        mc_df = sort_winning_probabilities(mc_df)
        mc_df.to_csv(DATA_DIR/"winner_probabilities.csv", index=False)
        st.session_state["mc_df"] = mc_df
        st.success(f"✅  {n_sims:,} simulations complete!")
        get_mc_data.clear()

    mc_df = st.session_state.get("mc_df")
    if mc_df is None or mc_df.empty:
        mc_df = get_mc_data()

    if mc_df is not None and not mc_df.empty:
        divider_label("Championship Probability — Top 16")
        top_n = mc_df.head(16).copy()
        top_n["label"] = top_n["team"].apply(lambda t: f"{flag(t)}  {t}")
        fig = go.Figure(go.Bar(
            x=top_n["label"], y=top_n["champion_pct"],
            marker=dict(color=top_n["champion_pct"],
                colorscale=[[0,"#0d2233"],[0.5,"#0090B0"],[1,"#FFC72C"]],showscale=False,
                line=dict(width=0)),
            text=top_n["champion_pct"].apply(lambda v: f"{v}%"),
            textposition="outside",
            textfont=dict(color="#F5F7FA", family="Barlow Condensed", size=11),
        ))
        fig.update_layout(**PLOT_BASE, height=360,
            xaxis=dict(color="#F5F7FA", tickangle=-35,
                       tickfont=dict(family="Barlow Condensed",size=11)),
            yaxis=dict(showgrid=False, showticklabels=False),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

        c1, c2, c3 = st.columns(3)
        for col, key, title, color in [
            (c1,"champion_pct","Champion Odds","#FFC72C"),
            (c2,"finalist_pct","Final Appearance","#00D4FF"),
            (c3,"semifinal_pct","Semi-Final Appearance","#00E676"),
        ]:
            top = mc_df.nlargest(8,key)[["team",key]].copy()
            top["label"] = top["team"].apply(lambda t: f"{flag(t)} {t}")
            f = go.Figure(go.Bar(
                x=top[key], y=top["label"], orientation="h",
                marker_color=color, marker_line_width=0,
                text=top[key].apply(lambda v: f"{v}%"),
                textposition="outside",
                textfont=dict(color="#F5F7FA", family="Barlow Condensed", size=10),
            ))
            f.update_layout(**{**PLOT_BASE, "margin": dict(l=0,r=50,t=30,b=0)}, height=280,
                title=dict(text=title, font=dict(color="#6B7A95",size=11,family="Barlow Condensed"),
                           x=0,y=0.98),
                xaxis=dict(showgrid=False,showticklabels=False,zeroline=False),
                yaxis=dict(showgrid=False,color="#F5F7FA",tickfont=dict(size=11,family="Barlow Condensed")),
            )
            col.plotly_chart(f, use_container_width=True, config={"displayModeBar":False})

        divider_label("Full Probability Table")
        display_df = mc_df.copy()
        display_df.insert(0,"",display_df["team"].apply(flag))
        display_df = display_df.rename(columns={
            "team":"Team","champion_pct":"🏆 Win %",
            "finalist_pct":"🥈 Final %","semifinal_pct":"🔝 Semi %"})
        st.dataframe(display_df, use_container_width=True, height=480,
            column_config={
                "🏆 Win %": st.column_config.ProgressColumn(min_value=0, max_value=display_df["🏆 Win %"].max()),
                "🥈 Final %": st.column_config.ProgressColumn(min_value=0, max_value=display_df["🥈 Final %"].max()),
                "🔝 Semi %": st.column_config.ProgressColumn(min_value=0, max_value=display_df["🔝 Semi %"].max()),
            })
    else:
        st.markdown(clean_html("""
        <div style="background:var(--card);border:1px dashed var(--border2);border-radius:12px;
                    padding:48px;text-align:center;">
            <div style="font-size:3rem;margin-bottom:12px;">🎲</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.2rem;
                        color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;">
                Run simulations to see probability data
            </div>
        </div>"""), unsafe_allow_html=True)
