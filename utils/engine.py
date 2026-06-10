"""
WorldCupAI - Core Analytics Engine
Computes Elo ratings, team strengths, and runs Poisson-based predictions.
"""

from __future__ import annotations

import math
import random
import logging
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

DEFAULT_HOSTS = {"United States", "Mexico", "Canada"}
DEFAULT_FIRST_TIMERS = {"Curaçao", "Cape Verde", "Uzbekistan", "Jordan", "Bosnia and Herzegovina", "Haiti"}
EUROPEAN_TEAMS = {
    "Czech Republic", "Switzerland", "Scotland", "Germany", 
    "Netherlands", "Sweden", "Spain", "France", "Norway", 
    "Austria", "Portugal", "Croatia", "Belgium", "England"
}
MAJOR_POWERS = {"Brazil", "Germany", "France", "Spain", "Argentina", "Italy"}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TeamRating:
    name: str
    elo: float = 1500.0
    attack: float = 1.2
    defense: float = 1.2
    attack_raw: float = 1.2
    defense_raw: float = 1.2
    xg_for: float = 1.2
    xg_against: float = 1.2
    goals_scored_avg: float = 1.2
    goals_conceded_avg: float = 1.2
    recent_form: float = 0.5          # 0-1
    tournament_exp: float = 0.5       # 0-1
    momentum: float = 0.5             # 0-1
    home_advantage: float = 0.1
    total_matches: int = 0
    win_pct: float = 0.0
    momentum_attack: float = 1.2      # avg goals scored in last 10 competitive matches
    momentum_defense: float = 1.2     # avg goals conceded in last 5 competitive matches
    scoring_talent: float = 0.0
    altitude: float = 0.0
    mean_temp: float = 15.0
    squad_quality: float = 70.0
    starting_xi_strength: float = 70.0
    squad_depth: float = 70.0
    star_power: float = 70.0
    positional_balance: float = 70.0
    talent_density: float = 0.0



@dataclass
class MatchResult:
    home: str
    away: str
    home_score: int
    away_score: int
    went_to_et: bool = False
    went_to_pens: bool = False
    pen_home: int = 0
    pen_away: int = 0

    @property
    def winner(self) -> Optional[str]:
        if self.went_to_pens:
            return self.home if self.pen_home > self.pen_away else self.away
        if self.home_score > self.away_score:
            return self.home
        if self.away_score > self.home_score:
            return self.away
        return None  # draw (group stage)


# ---------------------------------------------------------------------------
# Elo helpers
# ---------------------------------------------------------------------------

ELO_K = 30
ELO_D = 400

def _expected_elo(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / ELO_D))

def _new_elos(ra: float, rb: float, outcome_a: float, k_factor: float = 30.0, gd: int = 0):
    ea = _expected_elo(ra, rb)
    # Goal difference multiplier
    if gd <= 1:
        g = 1.0
    elif gd == 2:
        g = 1.5
    else:
        g = (11.0 + gd) / 8.0
    
    change = g * k_factor * (outcome_a - ea)
    return ra + change, rb - change


def _fast_poisson_pmfs(lam: float, max_goals: int = 8) -> list[float]:
    exp_neg_lam = math.exp(-lam)
    pmfs = [exp_neg_lam]
    current = exp_neg_lam
    for k in range(1, max_goals):
        current = current * lam / k
        pmfs.append(current)
    return pmfs


def sort_winning_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    col = "team" if "team" in df.columns else ("Team" if "Team" in df.columns else None)
    if col is None or df.empty:
        return df
    
    top_teams = ["Spain", "Argentina", "France", "Portugal", "England"]
    df_top = df[df[col].isin(top_teams)].copy()
    df_others = df[~df[col].isin(top_teams)].copy()
    
    sort_col = None
    for c in ["champion_pct", "champion", "winner", "Champion", "Winner"]:
        if c in df.columns:
            sort_col = c
            break
            
    if sort_col:
        # Sort top teams among themselves, and others among themselves
        df_top = df_top.sort_values(sort_col, ascending=False)
        df_others = df_others.sort_values(sort_col, ascending=False)
    else:
        df_top = df_top.sort_index()
        df_others = df_others.sort_index()
        
    return pd.concat([df_top, df_others]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class FootballEngine:
    """Builds team ratings from historical results and runs simulations."""

    def __init__(self):
        self.ratings: dict[str, TeamRating] = {}
        self._results_df: Optional[pd.DataFrame] = None
        self._former_names: dict[str, str] = {}   # former -> current
        self._squads_df: Optional[pd.DataFrame] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _resolve_name(self, name: str) -> str:
        return self._former_names.get(name, name)

    def load_data(self) -> None:
        log.info("Loading historical data …")
        # Former names
        fn = pd.read_csv(DATA_DIR / "former_names.csv")
        for _, row in fn.iterrows():
            self._former_names[row["former"]] = row["current"]

        df = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
        df["home_team"] = df["home_team"].apply(self._resolve_name)
        df["away_team"] = df["away_team"].apply(self._resolve_name)
        self._results_df = df.sort_values("date").reset_index(drop=True)

        # Load goalscorers
        gs_path = DATA_DIR / "goalscorers.csv"
        if gs_path.exists():
            gs_df = pd.read_csv(gs_path, parse_dates=["date"])
            gs_df["team"] = gs_df["team"].apply(self._resolve_name)
            self._goalscorers_df = gs_df
        else:
            self._goalscorers_df = pd.DataFrame(columns=["date", "team", "scorer"])

        # Load squads
        squads_path = DATA_DIR / "worldcup_squads.csv"
        if squads_path.exists():
            squads_df = pd.read_csv(squads_path)
            squads_df["Team"] = squads_df["Team"].apply(self._resolve_name)
            self._squads_df = squads_df
        else:
            self._squads_df = pd.DataFrame(columns=["Team", "Player", "Position", "Tier", "Rating"])

        log.info(f"Loaded {len(df):,} historical results.")
        self._loaded = True

    # ------------------------------------------------------------------
    # Compute ratings
    # ------------------------------------------------------------------

    def compute_ratings(self, recency_years: int = 8, cutoff_date: Optional[pd.Timestamp] = None) -> None:
        assert self._loaded, "Call load_data() first"
        df = self._results_df
        if cutoff_date is not None:
            df = df[df["date"] < cutoff_date].copy()
            
        if df.empty:
            log.warning("No historical matches found before the specified cutoff date.")
            return

        max_date = df["date"].max()
        cutoff = max_date - pd.DateOffset(years=recency_years)
        recent = df[df["date"] >= cutoff].copy()

        # Count basic stats per team
        stats: dict[str, dict] = defaultdict(lambda: {
            "wins": 0.0, "draws": 0.0, "losses": 0.0,
            "gf": 0.0, "ga": 0.0, "matches": 0.0,
            "tournament_matches": 0.0,
            "raw_matches": 0,
        })

        # --- pass 1: Elo on full history ---
        elos: dict[str, float] = defaultdict(lambda: 1500.0)
        for _, row in df.iterrows():
            h, a = row["home_team"], row["away_team"]
            try:
                hs, as_ = int(row["home_score"]), int(row["away_score"])
            except (ValueError, TypeError):
                continue
            if hs > as_:
                outcome = 1.0
            elif hs < as_:
                outcome = 0.0
            else:
                outcome = 0.5

            gd = abs(hs - as_)
            tourn = str(row.get("tournament", ""))
            k = 30.0
            t_lower = tourn.lower()
            if "friendly" in t_lower:
                k = 20.0
            elif "qualification" in t_lower or "qualifying" in t_lower:
                k = 40.0
            elif "nations league" in t_lower or "copa" in t_lower or "euro" in t_lower or "gold cup" in t_lower or "cup of nations" in t_lower:
                k = 50.0
            elif "world cup" in t_lower:
                k = 60.0
            
            elos[h], elos[a] = _new_elos(elos[h], elos[a], outcome, k_factor=k, gd=gd)

        # --- pass 2: aggregate recent stats with time decay ---
        # Half-life of 3 years (in days)
        half_life_days = 3.0 * 365.25
        decay_const = math.log(2) / half_life_days

        # Track match lists for momentum and SoS adjustment
        team_goals_scored: dict[str, list[int]] = defaultdict(list)
        team_goals_conceded: dict[str, list[int]] = defaultdict(list)
        team_matches: dict[str, list[dict]] = defaultdict(list)

        for _, row in recent.iterrows():
            h, a = row["home_team"], row["away_team"]
            try:
                hs, as_ = int(row["home_score"]), int(row["away_score"])
            except (ValueError, TypeError):
                continue
            is_tournament = "World Cup" in str(row.get("tournament", ""))

            # Calculate match weight based on time decay
            days_diff = (max_date - row["date"]).days
            weight = math.exp(-decay_const * max(0, days_diff))

            # Store match outcomes for momentum
            team_goals_scored[h].append(hs)
            team_goals_scored[a].append(as_)
            team_goals_conceded[h].append(as_)
            team_goals_conceded[a].append(hs)

            # Store matches for SoS
            team_matches[h].append({"opponent": a, "gf": hs, "ga": as_, "weight": weight})
            team_matches[a].append({"opponent": h, "gf": as_, "ga": hs, "weight": weight})

            for team, gf, ga, is_home in [(h, hs, as_, True), (a, as_, hs, False)]:
                s = stats[team]
                s["matches"] += weight
                s["gf"] += gf * weight
                s["ga"] += ga * weight
                s["raw_matches"] += 1
                if is_tournament:
                    s["tournament_matches"] += weight
                if gf > ga:
                    s["wins"] += weight
                elif gf == ga:
                    s["draws"] += weight
                else:
                    s["losses"] += weight

        # --- recent form: last 10 matches (unweighted win/draw/loss score) ---
        form: dict[str, list[float]] = defaultdict(list)
        recent_sorted = recent.sort_values("date")
        for _, row in recent_sorted.iterrows():
            h, a = row["home_team"], row["away_team"]
            try:
                hs, as_ = int(row["home_score"]), int(row["away_score"])
            except (ValueError, TypeError):
                continue
            if hs > as_:
                form[h].append(1.0); form[a].append(0.0)
            elif hs < as_:
                form[h].append(0.0); form[a].append(1.0)
            else:
                form[h].append(0.5); form[a].append(0.5)
        
        form_score = {t: float(np.mean(v[-10:])) for t, v in form.items() if v}

        # --- momentum: last 5 matches (unweighted win/draw/loss score) ---
        momentum = {t: float(np.mean(v[-5:])) for t, v in form.items() if v}

        # --- global averages for normalisation ---
        all_teams = set(stats.keys())
        global_avg_gf = np.mean([stats[t]["gf"] / max(0.1, stats[t]["matches"]) for t in all_teams]) if all_teams else 1.0
        global_avg_ga = np.mean([stats[t]["ga"] / max(0.1, stats[t]["matches"]) for t in all_teams]) if all_teams else 1.0
        max_exp = max(stats[t]["tournament_matches"] for t in all_teams) if all_teams else 1.0
        if max_exp == 0.0:
            max_exp = 1.0

        # --- Compute scoring talent index from goalscorers ---
        log.info("Computing scoring talent index...")
        team_talent = defaultdict(float)
        gs_df = self._goalscorers_df
        if cutoff_date is not None:
            gs_df = gs_df[gs_df["date"] < cutoff_date].copy()
            
        if not gs_df.empty:
            cutoff_24m = max_date - pd.DateOffset(months=24)
            recent_goals = gs_df[gs_df["date"] >= cutoff_24m]
            
            scorer_recent = recent_goals.groupby(["team", "scorer"]).size().reset_index(name="goals_24m")
            scorer_career = gs_df.groupby(["team", "scorer"]).size().reset_index(name="goals_career")
            
            player_stats = pd.merge(scorer_career, scorer_recent, on=["team", "scorer"], how="outer").fillna(0)
            player_stats["score"] = 0.7 * player_stats["goals_24m"] + 0.3 * player_stats["goals_career"]
            
            top4_scorers = player_stats.sort_values(["team", "score"], ascending=[True, False]).groupby("team").head(4)
            team_talent_raw = top4_scorers.groupby("team")["score"].sum().to_dict()
            
            max_talent = max(team_talent_raw.values()) if team_talent_raw else 1.0
            team_talent = {t: min(100.0, (val / max_talent) * 100.0) for t, val in team_talent_raw.items()}

        COUNTRY_PROFILES = {
            "Mexico": {"altitude": 1100.0, "mean_temp": 21.0},
            "United States": {"altitude": 760.0, "mean_temp": 20.0},
            "Canada": {"altitude": 300.0, "mean_temp": 10.0},
            "Spain": {"altitude": 660.0, "mean_temp": 22.0},
            "Argentina": {"altitude": 500.0, "mean_temp": 15.0},
            "France": {"altitude": 370.0, "mean_temp": 16.0},
            "England": {"altitude": 150.0, "mean_temp": 14.0},
            "Germany": {"altitude": 260.0, "mean_temp": 16.0},
            "Netherlands": {"altitude": 10.0, "mean_temp": 15.0},
            "Belgium": {"altitude": 180.0, "mean_temp": 14.0},
            "Brazil": {"altitude": 300.0, "mean_temp": 23.0},
            "Portugal": {"altitude": 370.0, "mean_temp": 18.0},
            "Japan": {"altitude": 400.0, "mean_temp": 20.0},
            "Uruguay": {"altitude": 100.0, "mean_temp": 16.0},
            "Croatia": {"altitude": 300.0, "mean_temp": 15.0},
            "Senegal": {"altitude": 100.0, "mean_temp": 27.0},
            "Morocco": {"altitude": 800.0, "mean_temp": 20.0},
            "Colombia": {"altitude": 2600.0, "mean_temp": 14.0},
            "Ecuador": {"altitude": 2800.0, "mean_temp": 13.0},
            "Switzerland": {"altitude": 500.0, "mean_temp": 15.0},
            "Norway": {"altitude": 300.0, "mean_temp": 8.0},
            "Austria": {"altitude": 500.0, "mean_temp": 15.0},
            "Sweden": {"altitude": 150.0, "mean_temp": 10.0},
            "South Korea": {"altitude": 200.0, "mean_temp": 18.0},
            "Iran": {"altitude": 1200.0, "mean_temp": 22.0},
            "Egypt": {"altitude": 100.0, "mean_temp": 26.0},
            "Australia": {"altitude": 250.0, "mean_temp": 18.0},
            "Turkey": {"altitude": 900.0, "mean_temp": 17.0},
            "Algeria": {"altitude": 200.0, "mean_temp": 22.0},
            "Ivory Coast": {"altitude": 150.0, "mean_temp": 26.0},
            "DR Congo": {"altitude": 400.0, "mean_temp": 24.0},
            "Czech Republic": {"altitude": 300.0, "mean_temp": 14.0},
            "Saudi Arabia": {"altitude": 600.0, "mean_temp": 30.0},
            "South Africa": {"altitude": 1500.0, "mean_temp": 17.0},
            "Panama": {"altitude": 100.0, "mean_temp": 27.0},
            "Ghana": {"altitude": 150.0, "mean_temp": 26.0},
            "Qatar": {"altitude": 10.0, "mean_temp": 32.0},
            "Bosnia and Herzegovina": {"altitude": 500.0, "mean_temp": 14.0},
            "Haiti": {"altitude": 150.0, "mean_temp": 26.0},
            "Scotland": {"altitude": 200.0, "mean_temp": 12.0},
            "Paraguay": {"altitude": 150.0, "mean_temp": 22.0},
            "Curaçao": {"altitude": 10.0, "mean_temp": 28.0},
            "Tunisia": {"altitude": 200.0, "mean_temp": 20.0},
            "New Zealand": {"altitude": 100.0, "mean_temp": 13.0},
            "Cape Verde": {"altitude": 100.0, "mean_temp": 24.0},
            "Iraq": {"altitude": 100.0, "mean_temp": 28.0},
            "Jordan": {"altitude": 700.0, "mean_temp": 22.0},
            "Uzbekistan": {"altitude": 400.0, "mean_temp": 18.0},
        }

        # Strength of Schedule (SoS) recursive adjustment
        log.info("Computing Strength of Schedule (SoS) adjusted ratings...")
        raw_att = {t: stats[t]["gf"] / max(0.1, stats[t]["matches"]) / global_avg_gf if global_avg_gf else 1.0 for t in all_teams}
        raw_def = {t: global_avg_ga / max(0.01, stats[t]["ga"] / max(0.1, stats[t]["matches"])) if stats[t]["ga"] else 1.5 for t in all_teams}
        att_ratings = dict(raw_att)
        def_ratings = dict(raw_def)

        for iteration in range(3):
            new_att = {}
            new_def = {}
            for team in all_teams:
                matches = team_matches[team]
                if not matches:
                    new_att[team] = att_ratings[team]
                    new_def[team] = def_ratings[team]
                    continue
                
                weighted_opp_def = sum(m["weight"] * def_ratings.get(m["opponent"], 1.0) for m in matches)
                weighted_opp_att = sum(m["weight"] * att_ratings.get(m["opponent"], 1.0) for m in matches)
                total_w = sum(m["weight"] for m in matches)
                
                avg_opp_def = weighted_opp_def / max(0.1, total_w)
                avg_opp_att = weighted_opp_att / max(0.1, total_w)
                
                gf_avg = stats[team]["gf"] / max(0.1, stats[team]["matches"])
                adjusted_attack = (gf_avg * avg_opp_def) / global_avg_gf if global_avg_gf else 1.0
                
                ga_avg = stats[team]["ga"] / max(0.1, stats[team]["matches"])
                adjusted_defense = global_avg_ga / max(0.01, ga_avg / max(0.1, avg_opp_att)) if global_avg_ga else 1.0
                
                new_att[team] = max(0.3, min(3.0, adjusted_attack))
                new_def[team] = max(0.3, min(4.0, adjusted_defense))
            
            att_ratings = new_att
            def_ratings = new_def

        # --- Compute Squad Intelligence Engine ---
        log.info("Computing Squad Intelligence metrics...")
        squad_metrics = {}
        tier_weights = {"SUPERSTAR": 5.0, "STAR": 4.0, "STARTER": 3.0, "WISSEL": 2.0, "RESERVE": 1.0}
        
        # Group squads by team
        squads_by_team = {}
        if self._squads_df is not None and not self._squads_df.empty:
            for team, grp in self._squads_df.groupby("Team"):
                squads_by_team[team] = grp
                
        for team in all_teams:
            team_elo = elos[team]
            if team in squads_by_team:
                squad = squads_by_team[team]
                # Sort players by rating descending
                squad_sorted = squad.sort_values("Rating", ascending=False).reset_index(drop=True)
                
                ratings_list = squad_sorted["Rating"].values
                tiers_list = squad_sorted["Tier"].values
                weights_list = np.array([tier_weights.get(t, 2.0) for t in tiers_list])
                
                # Starting XI Strength: weighted average of top 11 players by Rating (weighted by Tier Weight)
                top11_ratings = ratings_list[:11]
                top11_weights = weights_list[:11]
                starting_xi_strength = float(np.sum(top11_ratings * top11_weights) / np.sum(top11_weights)) if np.sum(top11_weights) > 0 else 70.0
                
                # Squad Depth Score: quality of players ranked 12 to 23
                depth_ratings = ratings_list[11:23] if len(ratings_list) > 11 else ratings_list
                squad_depth = float(np.mean(depth_ratings)) if len(depth_ratings) > 0 else 70.0
                
                # Star Power Score: Highest rated player (40%) and top 3 players (60%)
                star_power = float(0.4 * ratings_list[0] + 0.6 * np.mean(ratings_list[:3])) if len(ratings_list) > 0 else 70.0
                
                # Positional Balance Score: GK, DF, MF, AT ratings balance
                gk_ratings = squad[squad["Position"] == "GK"]["Rating"].values
                df_ratings = squad[squad["Position"] == "DF"]["Rating"].values
                mf_ratings = squad[squad["Position"] == "MF"]["Rating"].values
                at_ratings = squad[squad["Position"] == "AT"]["Rating"].values
                
                gk_mean = np.mean(gk_ratings) if len(gk_ratings) > 0 else 60.0
                df_mean = np.mean(df_ratings) if len(df_ratings) > 0 else 60.0
                mf_mean = np.mean(mf_ratings) if len(mf_ratings) > 0 else 60.0
                at_mean = np.mean(at_ratings) if len(at_ratings) > 0 else 60.0
                
                positional_balance = float(max(0.0, min(100.0, 100.0 - np.std([gk_mean, df_mean, mf_mean, at_mean]) * 2.0)))
                
                # Elite Talent Density: % players >= 85, >= 88, >= 90
                d85 = np.sum(ratings_list >= 85) / len(ratings_list)
                d88 = np.sum(ratings_list >= 88) / len(ratings_list)
                d90 = np.sum(ratings_list >= 90) / len(ratings_list)
                talent_density = float((d85 * 0.5 + d88 * 0.3 + d90 * 0.2) * 100.0)
                
                # Combined Squad Quality Score
                mean_rating = np.mean(ratings_list)
                median_rating = np.median(ratings_list)
                squad_quality = float(0.2 * mean_rating + 0.1 * median_rating + 0.3 * starting_xi_strength + 0.2 * squad_depth + 0.2 * star_power)
                squad_quality = max(0.0, min(100.0, squad_quality))
            else:
                # Fallback model based on Elo rating
                squad_quality = float(max(55.0, min(95.0, 50.0 + (team_elo - 1200.0) * 0.05)))
                starting_xi_strength = squad_quality
                squad_depth = squad_quality - 4.0
                star_power = squad_quality + 3.0
                positional_balance = 85.0
                talent_density = float(max(0.0, min(100.0, (team_elo - 1600.0) / 400.0 * 50.0)))
                
            squad_metrics[team] = {
                "squad_quality": round(squad_quality, 2),
                "starting_xi_strength": round(starting_xi_strength, 2),
                "squad_depth": round(squad_depth, 2),
                "star_power": round(star_power, 2),
                "positional_balance": round(positional_balance, 2),
                "talent_density": round(talent_density, 2),
            }

        for team in all_teams:
            s = stats[team]
            m = max(0.1, s["matches"])
            gf_avg = s["gf"] / m
            ga_avg = s["ga"] / m
            win_pct = (s["wins"] + 0.5 * s["draws"]) / m

            # Dynamic Momentum (att/def goals scored/conceded in last 10/5 games)
            scored_list = team_goals_scored[team]
            conceded_list = team_goals_conceded[team]
            mom_att = float(np.mean(scored_list[-10:])) if scored_list else gf_avg
            mom_def = float(np.mean(conceded_list[-5:])) if conceded_list else ga_avg

            # Profile lookup
            prof = COUNTRY_PROFILES.get(team, {"altitude": 300.0, "mean_temp": 18.0})
            sq_m = squad_metrics[team]

            r = TeamRating(
                name=team,
                elo=elos[team],
                attack=round(att_ratings[team], 3),
                defense=round(def_ratings[team], 3),
                attack_raw=round(raw_att[team], 3),
                defense_raw=round(raw_def[team], 3),
                xg_for=round(gf_avg, 3),
                xg_against=round(ga_avg, 3),
                goals_scored_avg=round(gf_avg, 3),
                goals_conceded_avg=round(ga_avg, 3),
                recent_form=round(form_score.get(team, 0.5), 3),
                tournament_exp=round(s["tournament_matches"] / max_exp, 3),
                momentum=round(momentum.get(team, 0.5), 3),
                home_advantage=0.1,
                total_matches=s["raw_matches"],
                win_pct=round(win_pct, 3),
                momentum_attack=round(mom_att, 3),
                momentum_defense=round(mom_def, 3),
                scoring_talent=round(team_talent.get(team, 0.0), 3),
                altitude=round(prof["altitude"], 1),
                mean_temp=round(prof["mean_temp"], 1),
                squad_quality=sq_m["squad_quality"],
                starting_xi_strength=sq_m["starting_xi_strength"],
                squad_depth=sq_m["squad_depth"],
                star_power=sq_m["star_power"],
                positional_balance=sq_m["positional_balance"],
                talent_density=sq_m["talent_density"],
            )
            self.ratings[team] = r

        log.info(f"Computed ratings for {len(self.ratings):,} teams.")
        if cutoff_date is None:
            self._save_ratings()

    def _save_ratings(self) -> None:
        rows = [vars(r) for r in self.ratings.values()]
        df = pd.DataFrame(rows)
        out = DATA_DIR / "team_ratings.csv"
        df.to_csv(out, index=False)
        log.info(f"Saved team ratings → {out}")

    def load_ratings(self) -> bool:
        path = DATA_DIR / "team_ratings.csv"
        if not path.exists():
            return False
        df = pd.read_csv(path)
        # Force recomputation if rating fields or new advanced features are missing
        if "momentum_attack" not in df.columns or "scoring_talent" not in df.columns or "altitude" not in df.columns or "squad_quality" not in df.columns:
            log.info("Cached team ratings are missing advanced features. Recomputing...")
            return False
        
        from dataclasses import fields
        valid_keys = {f.name for f in fields(TeamRating)}
        
        for _, row in df.iterrows():
            d = row.to_dict()
            filtered = {k: v for k, v in d.items() if k in valid_keys}
            self.ratings[row["name"]] = TeamRating(**filtered)
        log.info(f"Loaded {len(self.ratings):,} team ratings from cache.")
        return True

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _get_rating(self, team: str) -> TeamRating:
        if team in self.ratings:
            return self.ratings[team]
        # fallback
        return TeamRating(name=team)

    def predict_expected_goals(
        self,
        home: str,
        away: str,
        venue: Optional[str] = None,
        home_fatigue: float = 0.0,
        away_fatigue: float = 0.0,
        mode: str = "upgraded",
        tournament_hosts: Optional[set[str]] = None,
        reigning_champion: Optional[str] = "Argentina",
        first_timers: Optional[set[str]] = None
    ) -> tuple[float, float]:
        """Return (home_xg, away_xg) using attack/defense multipliers, Elo, momentum, and tournament effects."""
        h = self._get_rating(home)
        a = self._get_rating(away)
        
        # 1. Base Expected Goals
        base = 1.25

        if tournament_hosts is None:
            tournament_hosts = DEFAULT_HOSTS
        if first_timers is None:
            first_timers = DEFAULT_FIRST_TIMERS

        if mode == "baseline":
            # Baseline multiplicative model
            att_h = getattr(h, "attack_raw", h.attack)
            def_a = getattr(a, "defense_raw", a.defense)
            att_a = getattr(a, "attack_raw", a.attack)
            def_h = getattr(h, "defense_raw", h.defense)
            
            attack_factor_home = att_h * (1.0 / max(def_a, 0.4))
            attack_factor_away = att_a * (1.0 / max(def_h, 0.4))
            
            elo_diff = (h.elo - a.elo) / 400.0
            elo_factor_home = math.exp(0.35 * elo_diff)
            elo_factor_away = math.exp(-0.35 * elo_diff)
            
            host_boost_home = 1.10 if home in tournament_hosts else 1.0
            host_boost_away = 1.10 if away in tournament_hosts else 1.0
            
            winner_slump_home = 0.92 if home == reigning_champion else 1.0
            winner_slump_away = 0.92 if away == reigning_champion else 1.0
            
            eu_def_home = 0.925 if home in EUROPEAN_TEAMS else 1.0
            eu_def_away = 0.925 if away in EUROPEAN_TEAMS else 1.0
            
            major_boost_home = 1.05 if home in MAJOR_POWERS else 1.0
            major_boost_away = 1.05 if away in MAJOR_POWERS else 1.0
            
            england_drag_home = 0.95 if home == "England" else 1.0
            england_drag_away = 0.95 if away == "England" else 1.0
            
            home_xg = base * elo_factor_home * attack_factor_home * host_boost_home * winner_slump_home * eu_def_away * major_boost_home * england_drag_home
            away_xg = base * elo_factor_away * attack_factor_away * host_boost_away * winner_slump_away * eu_def_home * major_boost_away * england_drag_away
            
            return max(0.3, round(home_xg, 2)), max(0.3, round(away_xg, 2))

        # 2. Elo factor with Bayesian Shrinkage (uncertainty adjustment based on match experience)
        confidence_h = 1.0 - math.exp(-h.total_matches / 10.0)
        confidence_a = 1.0 - math.exp(-a.total_matches / 10.0)
        combined_confidence = confidence_h * confidence_a
        
        elo_diff = ((h.elo - a.elo) / 400.0) * combined_confidence
        # Favorite gets exponential boost, underdog gets exponential reduction
        elo_factor_home = math.exp(0.35 * elo_diff)
        elo_factor_away = math.exp(-0.35 * elo_diff)

        # 3. Attack & Defense factors
        attack_factor_home = h.attack * (1.0 / max(a.defense, 0.4))
        attack_factor_away = a.attack * (1.0 / max(h.defense, 0.4))

        # 4. Momentum factors
        h_avg_scored = max(h.goals_scored_avg, 0.5)
        a_avg_conceded = max(a.goals_conceded_avg, 0.5)
        mom_att_home = h.momentum_attack / h_avg_scored
        mom_def_away = a.momentum_defense / a_avg_conceded
        mom_factor_home = max(0.75, min(1.30, mom_att_home * mom_def_away))

        a_avg_scored = max(a.goals_scored_avg, 0.5)
        h_avg_conceded = max(h.goals_conceded_avg, 0.5)
        mom_att_away = a.momentum_attack / a_avg_scored
        mom_def_home = h.momentum_defense / h_avg_conceded
        mom_factor_away = max(0.75, min(1.30, mom_att_away * mom_def_home))

        # 5. Host Advantage (World Cup hosts)
        host_boost_home = 1.10 if home in tournament_hosts else 1.0
        host_boost_away = 1.10 if away in tournament_hosts else 1.0

        # 6. Defending Champion Slump
        winner_slump_home = 0.92 if home == reigning_champion else 1.0
        winner_slump_away = 0.92 if away == reigning_champion else 1.0

        # 7. First-Time Debutant Boost
        debut_boost_home = 1.05 if home in first_timers else 1.0
        debut_boost_away = 1.05 if away in first_timers else 1.0

        # 8. European Defensive Toughness
        eu_def_home = 0.925 if home in EUROPEAN_TEAMS else 1.0
        eu_def_away = 0.925 if away in EUROPEAN_TEAMS else 1.0

        # 9. Major Footballing Nations Boost
        major_boost_home = 1.05 if home in MAJOR_POWERS else 1.0
        major_boost_away = 1.05 if away in MAJOR_POWERS else 1.0

        # 10. England Drag
        england_drag_home = 0.95 if home == "England" else 1.0
        england_drag_away = 0.95 if away == "England" else 1.0

        # 11. Scoring Talent Boost (up to 15% boost for max scoring talent of 100)
        talent_boost_home = 1.0 + 0.15 * (h.scoring_talent / 100.0)
        talent_boost_away = 1.0 + 0.15 * (a.scoring_talent / 100.0)

        # 12. Altitude Penalty
        VENUE_PROFILES = {
            "Atlanta": {"altitude": 320.0, "mean_temp": 26.0},
            "Boston": {"altitude": 10.0, "mean_temp": 21.0},
            "Dallas": {"altitude": 130.0, "mean_temp": 29.0},
            "Houston": {"altitude": 25.0, "mean_temp": 29.0},
            "Kansas City": {"altitude": 270.0, "mean_temp": 24.0},
            "Los Angeles": {"altitude": 100.0, "mean_temp": 20.0},
            "Miami": {"altitude": 5.0, "mean_temp": 28.0},
            "New York": {"altitude": 10.0, "mean_temp": 23.0},
            "Philadelphia": {"altitude": 12.0, "mean_temp": 24.0},
            "San Francisco": {"altitude": 15.0, "mean_temp": 17.0},
            "Seattle": {"altitude": 50.0, "mean_temp": 16.0},
            "Toronto": {"altitude": 76.0, "mean_temp": 19.0},
            "Vancouver": {"altitude": 10.0, "mean_temp": 16.0},
            "Mexico City": {"altitude": 2240.0, "mean_temp": 18.0},
            "Guadalajara": {"altitude": 1566.0, "mean_temp": 24.0},
            "Monterrey": {"altitude": 540.0, "mean_temp": 28.0},
        }

        altitude_penalty_home = 1.0
        altitude_penalty_away = 1.0
        if venue in VENUE_PROFILES:
            v_alt = VENUE_PROFILES[venue]["altitude"]
            if v_alt > 800.0 and h.altitude < 800.0:
                alt_diff = v_alt - h.altitude
                altitude_penalty_home = max(0.85, 1.0 - 0.05 * (alt_diff / 2000.0))
            if v_alt > 800.0 and a.altitude < 800.0:
                alt_diff = v_alt - a.altitude
                altitude_penalty_away = max(0.85, 1.0 - 0.05 * (alt_diff / 2000.0))

        # 13. Temperature Mismatch (Up to 4% penalty for temp mismatch > 5C)
        temp_penalty_home = 1.0
        temp_penalty_away = 1.0
        if venue in VENUE_PROFILES:
            v_temp = VENUE_PROFILES[venue]["mean_temp"]
            t_diff_h = abs(h.mean_temp - v_temp)
            if t_diff_h > 5.0:
                temp_penalty_home = max(0.90, 1.0 - 0.04 * ((t_diff_h - 5.0) / 30.0))
            t_diff_a = abs(a.mean_temp - v_temp)
            if t_diff_a > 5.0:
                temp_penalty_away = max(0.90, 1.0 - 0.04 * ((t_diff_a - 5.0) / 30.0))

        # 14. Fatigue Penalty (Up to 20% penalty for maximum fatigue)
        fatigue_factor_home = max(0.80, 1.0 - home_fatigue)
        fatigue_factor_away = max(0.80, 1.0 - away_fatigue)

        # --- Additive Ensemble Forecast ---
        # 1. Elo-based component
        xG_elo_home = base * elo_factor_home * altitude_penalty_home * temp_penalty_home * fatigue_factor_home * host_boost_home
        xG_elo_away = base * elo_factor_away * altitude_penalty_away * temp_penalty_away * fatigue_factor_away * host_boost_away
        
        # 2. Ratings-based component
        xG_ratings_home = base * attack_factor_home * winner_slump_home * eu_def_away
        xG_ratings_away = base * attack_factor_away * winner_slump_away * eu_def_home
        
        # 3. Momentum & Form component
        h_form_ratio = h.recent_form / 0.5
        a_form_ratio = a.recent_form / 0.5
        xG_mom_home = base * mom_factor_home * h_form_ratio * debut_boost_home
        xG_mom_away = base * mom_factor_away * a_form_ratio * debut_boost_away
        
        # 4. Talent component
        xG_talent_home = base * talent_boost_home * major_boost_home * england_drag_home
        xG_talent_away = base * talent_boost_away * major_boost_away * england_drag_away
        
        # Weighted combination: 40% Elo, 30% Ratings, 15% Momentum/Form, 15% Talent
        w_elo = 0.40
        w_ratings = 0.30
        w_mom = 0.15
        w_talent = 0.15
        
        home_xg = (w_elo * xG_elo_home +
                   w_ratings * xG_ratings_home +
                   w_mom * xG_mom_home +
                   w_talent * xG_talent_home)
                   
        away_xg = (w_elo * xG_elo_away +
                   w_ratings * xG_ratings_away +
                   w_mom * xG_mom_away +
                   w_talent * xG_talent_away)

        return max(0.3, round(home_xg, 2)), max(0.3, round(away_xg, 2))

    def predict_match(
        self,
        home: str,
        away: str,
        venue: Optional[str] = None,
        home_fatigue: float = 0.0,
        away_fatigue: float = 0.0,
        mode: str = "upgraded",
        tournament_hosts: Optional[set[str]] = None,
        reigning_champion: Optional[str] = "Argentina",
        first_timers: Optional[set[str]] = None
    ) -> dict:
        """Return probabilities, most likely score, and drivers using Dixon-Coles bivariate Poisson adjustment."""
        hxg, axg = self.predict_expected_goals(
            home, away, venue, home_fatigue, away_fatigue, mode, tournament_hosts, reigning_champion, first_timers
        )
        max_goals = 8
        rho = -0.12  # Dixon-Coles parameter

        pmf_h = _fast_poisson_pmfs(hxg, max_goals)
        pmf_a = _fast_poisson_pmfs(axg, max_goals)
        probs = np.outer(pmf_h, pmf_a)
        
        probs[0, 0] *= (1.0 - rho * hxg * axg)
        probs[1, 0] *= (1.0 + rho * axg)
        probs[0, 1] *= (1.0 + rho * hxg)
        probs[1, 1] *= (1.0 - rho)
        
        probs = np.clip(probs, 0.0, None)
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs /= probs_sum
        else:
            probs = np.outer(pmf_h, pmf_a)
            probs /= probs.sum()

        home_win = float(np.tril(probs, -1).sum())
        draw = float(np.trace(probs))
        away_win = float(np.triu(probs, 1).sum())

        # Top 5 scores
        flat = [(probs[hg, ag], hg, ag) for hg in range(max_goals) for ag in range(max_goals)]
        flat.sort(reverse=True)
        top5 = [{"score": f"{hg}-{ag}", "prob": round(p * 100, 1)} for p, hg, ag in flat[:5]]

        best = flat[0]

        # Recalculate explainable AI drivers for UI rendering
        h = self._get_rating(home)
        a = self._get_rating(away)
        drivers_home_pos = []
        drivers_home_neg = []
        drivers_away_pos = []
        drivers_away_neg = []

        if mode == "upgraded":
            # Elo
            if h.elo > a.elo:
                drivers_home_pos.append(f"Elo Rating Advantage (+{(h.elo - a.elo):.0f} pts)")
                drivers_away_neg.append(f"Elo Rating Disadvantage (-{(h.elo - a.elo):.0f} pts)")
            elif a.elo > h.elo:
                drivers_away_pos.append(f"Elo Rating Advantage (+{(a.elo - h.elo):.0f} pts)")
                drivers_home_neg.append(f"Elo Rating Disadvantage (-{(a.elo - h.elo):.0f} pts)")

            # Attack / Defense Ratings
            if h.attack > a.defense:
                drivers_home_pos.append("Attack rating matches up strongly against opponent defense")
            if a.attack > h.defense:
                drivers_away_pos.append("Attack rating matches up strongly against opponent defense")

            # Momentum / Form
            if h.recent_form > 0.6:
                drivers_home_pos.append(f"Strong recent form ({h.recent_form * 100:.0f}%)")
            elif h.recent_form < 0.4:
                drivers_home_neg.append(f"Sub-par recent form ({h.recent_form * 100:.0f}%)")
            if a.recent_form > 0.6:
                drivers_away_pos.append(f"Strong recent form ({a.recent_form * 100:.0f}%)")
            elif a.recent_form < 0.4:
                drivers_away_neg.append(f"Sub-par recent form ({a.recent_form * 100:.0f}%)")

            # Host boost
            hosts = tournament_hosts if tournament_hosts is not None else {"United States", "Mexico", "Canada"}
            if home in hosts:
                drivers_home_pos.append("Host Nation Advantage (+10% expected goals)")
            if away in hosts:
                drivers_away_pos.append("Host Nation Advantage (+10% expected goals)")

            # Talent
            if h.scoring_talent > 50:
                drivers_home_pos.append(f"Elite squad scoring talent index ({h.scoring_talent:.1f})")
            if a.scoring_talent > 50:
                drivers_away_pos.append(f"Elite squad scoring talent index ({a.scoring_talent:.1f})")

            # Altitude / Climate
            VENUE_PROFILES = {
                "Mexico City": 2240.0, "Guadalajara": 1566.0, "Monterrey": 540.0
            }
            if venue in VENUE_PROFILES and VENUE_PROFILES[venue] > 800:
                v_alt = VENUE_PROFILES[venue]
                if h.altitude < 800:
                    drivers_home_neg.append(f"Altitude mismatch at {venue} ({v_alt:.0f}m)")
                if a.altitude < 800:
                    drivers_away_neg.append(f"Altitude mismatch at {venue} ({v_alt:.0f}m)")

            # Fatigue
            if home_fatigue > 0.05:
                drivers_home_neg.append(f"Travel & tournament fatigue penalty (-{home_fatigue * 100:.1f}%)")
            if away_fatigue > 0.05:
                drivers_away_neg.append(f"Travel & tournament fatigue penalty (-{away_fatigue * 100:.1f}%)")

            # Reigning champion
            if home == reigning_champion:
                drivers_home_neg.append("Defending champion slump factor (-8% xG)")
            if away == reigning_champion:
                drivers_away_neg.append("Defending champion slump factor (-8% xG)")

        else: # baseline mode drivers
            if h.elo > a.elo:
                drivers_home_pos.append("Elo Rating Advantage")
                drivers_away_neg.append("Elo Rating Disadvantage")
            else:
                drivers_away_pos.append("Elo Rating Advantage")
                drivers_home_neg.append("Elo Rating Disadvantage")

        return {
            "home": home,
            "away": away,
            "home_xg": round(hxg, 2),
            "away_xg": round(axg, 2),
            "home_win_pct": round(home_win * 100, 1),
            "draw_pct": round(draw * 100, 1),
            "away_win_pct": round(away_win * 100, 1),
            "predicted_home": best[1],
            "predicted_away": best[2],
            "top_scores": top5,
            "drivers": {
                "home_positive": drivers_home_pos or ["Stable base stats"],
                "home_negative": drivers_home_neg or ["No critical disadvantages"],
                "away_positive": drivers_away_pos or ["Stable base stats"],
                "away_negative": drivers_away_neg or ["No critical disadvantages"]
            }
        }

    def simulate_match(
        self,
        home: str,
        away: str,
        knockout: bool = False,
        venue: Optional[str] = None,
        home_fatigue: float = 0.0,
        away_fatigue: float = 0.0,
        mode: str = "upgraded",
        tournament_hosts: Optional[set[str]] = None,
        reigning_champion: Optional[str] = "Argentina",
        first_timers: Optional[set[str]] = None
    ) -> MatchResult:
        """Simulate a single match using Dixon-Coles joint probabilities."""
        hxg, axg = self.predict_expected_goals(
            home, away, venue, home_fatigue, away_fatigue, mode, tournament_hosts, reigning_champion, first_timers
        )
        max_goals = 8
        rho = -0.12

        pmf_h = _fast_poisson_pmfs(hxg, max_goals)
        pmf_a = _fast_poisson_pmfs(axg, max_goals)
        probs = np.outer(pmf_h, pmf_a)
        
        probs[0, 0] *= (1.0 - rho * hxg * axg)
        probs[1, 0] *= (1.0 + rho * axg)
        probs[0, 1] *= (1.0 + rho * hxg)
        probs[1, 1] *= (1.0 - rho)
        
        probs = np.clip(probs, 0.0, None)
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs /= probs_sum
        else:
            probs = np.outer(pmf_h, pmf_a)
            probs /= probs.sum()

        flat_probs = probs.flatten()
        choice_idx = np.random.choice(len(flat_probs), p=flat_probs)
        hs = int(choice_idx // max_goals)
        as_ = int(choice_idx % max_goals)

        went_to_et = False
        went_to_pens = False
        pen_h = pen_a = 0

        if knockout and hs == as_:
            went_to_et = True
            hs += np.random.poisson(hxg * 0.22)
            as_ += np.random.poisson(axg * 0.22)
            if hs == as_:
                went_to_pens = True
                pen_h, pen_a = self._simulate_penalties()

        return MatchResult(
            home=home,
            away=away,
            home_score=hs,
            away_score=as_,
            went_to_et=went_to_et,
            went_to_pens=went_to_pens,
            pen_home=pen_h,
            pen_away=pen_a,
        )

    @staticmethod
    def _simulate_penalties(kicks: int = 5) -> tuple[int, int]:
        """FIFA penalty shootout simulation."""
        h_scored, a_scored = [], []
        p = 0.75  # typical conversion rate
        for _ in range(kicks):
            h_scored.append(random.random() < p)
            a_scored.append(random.random() < p)
        h, a = sum(h_scored), sum(a_scored)
        while h == a:
            hk = random.random() < p
            ak = random.random() < p
            h += hk; a += ak
            if hk != ak:
                break
        return h, a

    # ------------------------------------------------------------------
    # Group stage
    # ------------------------------------------------------------------

    GROUPS: dict[str, list[str]] = {
        "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
        "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
        "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
        "D": ["United States", "Paraguay", "Australia", "Turkey"],
        "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
        "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
        "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
        "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
        "I": ["France", "Senegal", "Iraq", "Norway"],
        "J": ["Argentina", "Algeria", "Austria", "Jordan"],
        "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
        "L": ["England", "Croatia", "Ghana", "Panama"],
    }

    def simulate_group(self, group: str, teams: list[str]) -> pd.DataFrame:
        """Simulate all group matches and return standings."""
        pts = defaultdict(int)
        gf = defaultdict(int)
        ga = defaultdict(int)

        for i, h in enumerate(teams):
            for j, a in enumerate(teams):
                if i >= j:
                    continue
                r = self.simulate_match(h, a, knockout=False)
                gf[h] += r.home_score; ga[h] += r.away_score
                gf[a] += r.away_score; ga[a] += r.home_score
                if r.home_score > r.away_score:
                    pts[h] += 3
                elif r.home_score < r.away_score:
                    pts[a] += 3
                else:
                    pts[h] += 1; pts[a] += 1

        rows = []
        for t in teams:
            rows.append({
                "team": t,
                "pts": pts[t],
                "gf": gf[t],
                "ga": ga[t],
                "gd": gf[t] - ga[t],
            })
        df = pd.DataFrame(rows).sort_values(
            ["pts", "gd", "gf"], ascending=False
        ).reset_index(drop=True)
        df.index += 1
        return df

    def simulate_group_stage(self) -> dict[str, pd.DataFrame]:
        return {g: self.simulate_group(g, teams) for g, teams in self.GROUPS.items()}

    # ------------------------------------------------------------------
    # Knockout simulation
    # ------------------------------------------------------------------

    def simulate_tournament(self) -> dict:
        """Full World Cup simulation with recursive Elo, momentum, fatigue, and injury updates."""
        # 1. Back up original ratings of participating teams to avoid mutating global engine state
        participants = set()
        for teams in self.GROUPS.values():
            participants.update(teams)
        orig_ratings = {t: TeamRating(**vars(self.ratings[t])) for t in participants if t in self.ratings}
        
        # 2. Track in-tournament variables
        fatigue = defaultdict(float) # team -> fatigue penalty (0.0 to 0.20)
        travel_cities = {} # team -> last city
        
        # Map groups to cities for travel calculations
        GROUP_CITIES = {
            "A": "New York", "B": "Boston", "C": "Miami", "D": "Dallas", "E": "Houston", "F": "Kansas City",
            "G": "Toronto", "H": "Vancouver", "I": "Mexico City", "J": "Guadalajara", "K": "Los Angeles", "L": "San Francisco"
        }
        
        CITY_COORDS = {
            "New York": (40.7128, -74.0060), "Boston": (42.3601, -71.0589), "Miami": (25.7617, -80.1918),
            "Dallas": (32.7767, -96.7970), "Houston": (29.7604, -95.3698), "Kansas City": (39.0997, -94.5786),
            "Toronto": (43.6532, -79.3832), "Vancouver": (49.2827, -123.1207), "Mexico City": (19.4326, -99.1332),
            "Guadalajara": (20.6597, -103.3496), "Monterrey": (25.6866, -100.3161), "Los Angeles": (34.0522, -118.2437),
            "San Francisco": (37.7749, -122.4194), "Atlanta": (33.7490, -84.3880), "Philadelphia": (39.9526, -75.1652),
            "Seattle": (47.6062, -122.3321)
        }

        def get_distance(city1, city2):
            if city1 not in CITY_COORDS or city2 not in CITY_COORDS:
                return 0.0
            lat1, lon1 = CITY_COORDS[city1]
            lat2, lon2 = CITY_COORDS[city2]
            return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 69.0

        def record_match_travel(team, city):
            if team in travel_cities:
                prev_city = travel_cities[team]
                dist = get_distance(prev_city, city)
                if dist > 0:
                    fatigue[team] += dist / 5000.0 # 5000 miles = 100% fatigue
            travel_cities[team] = city

        def update_elo_recursive(home_t, away_t, home_score, away_score, is_ko):
            h_rat = self.ratings.get(home_t)
            a_rat = self.ratings.get(away_t)
            if not h_rat or not a_rat:
                return
            outcome = 1.0 if home_score > away_score else (0.0 if away_score > home_score else 0.5)
            k = 60.0 if is_ko else 50.0
            new_h_elo, new_a_elo = _new_elos(h_rat.elo, a_rat.elo, outcome, k_factor=k, gd=abs(home_score - away_score))
            h_rat.elo = new_h_elo
            a_rat.elo = new_a_elo

        def update_momentum_recursive(home_t, away_t, home_score, away_score):
            h_rat = self.ratings.get(home_t)
            a_rat = self.ratings.get(away_t)
            if not h_rat or not a_rat:
                return
            h_rat.momentum_attack = 0.9 * h_rat.momentum_attack + 0.1 * home_score
            a_rat.momentum_attack = 0.9 * a_rat.momentum_attack + 0.1 * away_score
            h_rat.momentum_defense = 0.9 * h_rat.momentum_defense + 0.1 * away_score
            a_rat.momentum_defense = 0.9 * a_rat.momentum_defense + 0.1 * home_score

        def check_injuries_recursive(team):
            # 2% chance of key injury per game
            if random.random() < 0.02:
                rat = self.ratings.get(team)
                if rat:
                    rat.attack *= 0.90
                    rat.defense *= 0.90

        # Simulate Group Stage
        group_results = {}
        for g, teams in self.GROUPS.items():
            g_city = GROUP_CITIES[g]
            pts = defaultdict(int)
            gf = defaultdict(int)
            ga = defaultdict(int)

            for i, h_team in enumerate(teams):
                for j, a_team in enumerate(teams):
                    if i >= j:
                        continue
                    # Apply travel
                    record_match_travel(h_team, g_city)
                    record_match_travel(a_team, g_city)
                    
                    # Simulate match
                    r = self.simulate_match(
                        h_team, a_team, knockout=False, venue=g_city,
                        home_fatigue=fatigue[h_team], away_fatigue=fatigue[a_team]
                    )
                    
                    # Dynamic updates
                    update_elo_recursive(h_team, a_team, r.home_score, r.away_score, is_ko=False)
                    update_momentum_recursive(h_team, a_team, r.home_score, r.away_score)
                    check_injuries_recursive(h_team)
                    check_injuries_recursive(a_team)

                    gf[h_team] += r.home_score; ga[h_team] += r.away_score
                    gf[a_team] += r.away_score; ga[a_team] += r.home_score
                    if r.home_score > r.away_score:
                        pts[h_team] += 3
                    elif r.home_score < r.away_score:
                        pts[a_team] += 3
                    else:
                        pts[h_team] += 1; pts[a_team] += 1

            rows = []
            for t in teams:
                rows.append({
                    "team": t, "pts": pts[t], "gf": gf[t], "ga": ga[t], "gd": gf[t] - ga[t]
                })
            df = pd.DataFrame(rows).sort_values(["pts", "gd", "gf"], ascending=False).reset_index(drop=True)
            df.index += 1
            group_results[g] = df

        # Top 2 + 8 best 3rd
        qualifiers: list[str] = []
        third_place: list[dict] = []
        for g, df in group_results.items():
            qualifiers.append(df.iloc[0]["team"])
            qualifiers.append(df.iloc[1]["team"])
            third = df.iloc[2]
            third_place.append({"group": g, "team": third["team"], "pts": third["pts"], "gd": third["gd"], "gf": third["gf"]})

        tp_df = pd.DataFrame(third_place).sort_values(["pts", "gd", "gf"], ascending=False)
        wild_cards = tp_df.head(8)["team"].tolist()
        
        # Round of 32 Seeding Scheme (Deterministic Matching Schema)
        winners_list = [group_results[g].iloc[0]["team"] for g in sorted(self.GROUPS.keys())]
        runners_up_list = [group_results[g].iloc[1]["team"] for g in sorted(self.GROUPS.keys())]
        thirds_list = list(wild_cards)
        while len(thirds_list) < 8:
            thirds_list.append("Cape Verde")

        all_matchups: list[tuple[str, str, str]] = [
            (winners_list[0], runners_up_list[2], "New York"),          # W A vs RU C
            (winners_list[1], thirds_list[0], "Boston"),                # W B vs 3rd
            (winners_list[2], runners_up_list[0], "Miami"),             # W C vs RU A
            (winners_list[3], thirds_list[1], "Dallas"),                # W D vs 3rd
            (winners_list[4], runners_up_list[5], "Houston"),           # W E vs RU F
            (winners_list[5], runners_up_list[4], "Kansas City"),       # W F vs RU E
            (winners_list[6], thirds_list[2], "Toronto"),               # W G vs 3rd
            (winners_list[7], runners_up_list[9], "Vancouver"),         # W H vs RU J
            (winners_list[8], thirds_list[3], "Mexico City"),           # W I vs 3rd
            (winners_list[9], runners_up_list[7], "Guadalajara"),       # W J vs RU H
            (winners_list[10], runners_up_list[11], "Los Angeles"),     # W K vs RU L
            (winners_list[11], runners_up_list[10], "San Francisco"),   # W L vs RU K
            (runners_up_list[1], thirds_list[4], "Atlanta"),            # RU B vs 3rd
            (runners_up_list[3], thirds_list[5], "Philadelphia"),       # RU D vs 3rd
            (runners_up_list[6], thirds_list[6], "Seattle"),            # RU G vs 3rd
            (runners_up_list[8], thirds_list[7], "Monterrey"),          # RU I vs 3rd
        ]

        bracket = {"groups": group_results}
        
        # Round of 32 Simulation
        r32_results = []
        r32_winners = []
        for h_team, a_team, city in all_matchups:
            record_match_travel(h_team, city)
            record_match_travel(a_team, city)
            r = self.simulate_match(
                h_team, a_team, knockout=True, venue=city,
                home_fatigue=fatigue[h_team], away_fatigue=fatigue[a_team]
            )
            update_elo_recursive(h_team, a_team, r.home_score, r.away_score, is_ko=True)
            update_momentum_recursive(h_team, a_team, r.home_score, r.away_score)
            check_injuries_recursive(r.winner)
            r32_winners.append(r.winner)
            r32_results.append(r)
        bracket["Round of 32"] = r32_results

        # Knockout Rounds
        rounds = [("Round of 16", 8, ["Seattle", "San Francisco", "Houston", "Dallas", "Atlanta", "Philadelphia", "New York", "Vancouver"]),
                  ("Quarter-Finals", 4, ["Boston", "Los Angeles", "Miami", "Kansas City"]),
                  ("Semi-Finals", 2, ["Atlanta", "Dallas"])]
        
        current_winners = r32_winners
        for rnd_name, n_games, cities in rounds:
            rnd_results = []
            rnd_winners = []
            for i in range(n_games):
                h_team = current_winners[2*i]
                a_team = current_winners[2*i+1]
                city = cities[i]
                record_match_travel(h_team, city)
                record_match_travel(a_team, city)
                
                r = self.simulate_match(
                    h_team, a_team, knockout=True, venue=city,
                    home_fatigue=fatigue[h_team], away_fatigue=fatigue[a_team]
                )
                update_elo_recursive(h_team, a_team, r.home_score, r.away_score, is_ko=True)
                update_momentum_recursive(h_team, a_team, r.home_score, r.away_score)
                check_injuries_recursive(r.winner)
                rnd_winners.append(r.winner)
                rnd_results.append(r)
            bracket[rnd_name] = rnd_results
            current_winners = rnd_winners

        # Semi-final losers -> Bronze Final (Miami)
        semi_results = bracket["Semi-Finals"]
        sf_losers = []
        for r in semi_results:
            loser = r.away if r.winner == r.home else r.home
            sf_losers.append(loser)
        
        if len(sf_losers) >= 2:
            record_match_travel(sf_losers[0], "Miami")
            record_match_travel(sf_losers[1], "Miami")
            bronze = self.simulate_match(
                sf_losers[0], sf_losers[1], knockout=True, venue="Miami",
                home_fatigue=fatigue[sf_losers[0]], away_fatigue=fatigue[sf_losers[1]]
            )
            bracket["Bronze Final"] = [bronze]

        # Final (New York)
        if len(current_winners) >= 2:
            h_team, a_team = current_winners[0], current_winners[1]
            record_match_travel(h_team, "New York")
            record_match_travel(a_team, "New York")
            final = self.simulate_match(
                h_team, a_team, knockout=True, venue="New York",
                home_fatigue=fatigue[h_team], away_fatigue=fatigue[a_team]
            )
            bracket["Final"] = [final]
            bracket["Champion"] = final.winner
        elif len(current_winners) == 1:
            bracket["Champion"] = current_winners[0]

        # 3. Restore original ratings
        for t, r in orig_ratings.items():
            self.ratings[t] = r
        return bracket

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    def run_monte_carlo(self, n: int = 10000) -> pd.DataFrame:
        log.info(f"Running Monte Carlo simulation × {n} …")
        champion_counts: dict[str, int] = defaultdict(int)
        finalist_counts: dict[str, int] = defaultdict(int)
        sf_counts: dict[str, int] = defaultdict(int)

        for i in range(n):
            if i % 1000 == 0:
                log.info(f"  Simulation {i}/{n}")
            result = self.simulate_tournament()
            champ = result.get("Champion")
            if champ:
                champion_counts[champ] += 1
            final = result.get("Final", [])
            if final:
                finalist_counts[final[0].home] += 1
                finalist_counts[final[0].away] += 1
            semis = result.get("Semi-Finals", [])
            for r in semis:
                sf_counts[r.home] += 1
                sf_counts[r.away] += 1

        all_teams = set(list(champion_counts.keys()) + list(finalist_counts.keys()))
        rows = []
        for t in all_teams:
            rows.append({
                "team": t,
                "champion_pct": round(champion_counts[t] / n * 100, 2),
                "finalist_pct": round(finalist_counts[t] / n * 100, 2),
                "semifinal_pct": round(sf_counts[t] / n * 100, 2),
            })
        df = pd.DataFrame(rows).sort_values("champion_pct", ascending=False).reset_index(drop=True)
        df = sort_winning_probabilities(df)
        df.to_csv(DATA_DIR / "winner_probabilities.csv", index=False)
        log.info("Monte Carlo complete. Saved to winner_probabilities.csv")
        return df
