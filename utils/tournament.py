"""
WorldCupAI — Tournament Predictor Engine
Parses the official FIFA 2026 fixture schedule and runs a complete
match-by-match simulation following the real fixture order.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ── Fixture data (parsed from the official PDF schedule) ──────────────────
# Format: (date_str, group_or_round, home_code, away_code, matchday)
# Codes are mapped to full names via TEAM_CODES below.

TEAM_CODES: dict[str, str] = {
    "MEX": "Mexico",        "RSA": "South Africa",   "KOR": "South Korea",
    "CZE": "Czech Republic","CAN": "Canada",          "BIH": "Bosnia and Herzegovina",
    "QAT": "Qatar",         "SUI": "Switzerland",     "BRA": "Brazil",
    "MAR": "Morocco",       "HAI": "Haiti",           "SCO": "Scotland",
    "USA": "United States", "PAR": "Paraguay",        "AUS": "Australia",
    "TUR": "Turkey",        "GER": "Germany",         "CUW": "Curaçao",
    "CIV": "Ivory Coast",   "ECU": "Ecuador",         "NED": "Netherlands",
    "JPN": "Japan",         "SWE": "Sweden",          "TUN": "Tunisia",
    "BEL": "Belgium",       "EGY": "Egypt",           "IRN": "Iran",
    "NZL": "New Zealand",   "ESP": "Spain",           "CPV": "Cape Verde",
    "KSA": "Saudi Arabia",  "URU": "Uruguay",         "FRA": "France",
    "SEN": "Senegal",       "IRQ": "Iraq",            "NOR": "Norway",
    "ARG": "Argentina",     "ALG": "Algeria",         "AUT": "Austria",
    "JOR": "Jordan",        "POR": "Portugal",        "COD": "DR Congo",
    "UZB": "Uzbekistan",    "COL": "Colombia",        "ENG": "England",
    "CRO": "Croatia",       "GHA": "Ghana",           "PAN": "Panama",
}

# Raw fixtures from the PDF: (date, round, group, home_code, away_code, matchday)
# matchday 1/2/3 = group stage matchdays; 0 = knockout
_RAW_FIXTURES = [
    # ── Matchday 1 ─────────────────────────────────────────────────────
    ("Jun 12", "Group Stage", "A", "MEX", "RSA",  1),
    ("Jun 12", "Group Stage", "A", "KOR", "CZE",  1),
    ("Jun 13", "Group Stage", "B", "CAN", "BIH",  1),
    ("Jun 13", "Group Stage", "D", "USA", "PAR",  1),
    ("Jun 14", "Group Stage", "B", "QAT", "SUI",  1),
    ("Jun 14", "Group Stage", "C", "BRA", "MAR",  1),
    ("Jun 14", "Group Stage", "C", "HAI", "SCO",  1),
    ("Jun 14", "Group Stage", "D", "AUS", "TUR",  1),
    ("Jun 15", "Group Stage", "E", "GER", "CUW",  1),
    ("Jun 15", "Group Stage", "F", "NED", "JPN",  1),
    ("Jun 15", "Group Stage", "E", "CIV", "ECU",  1),
    ("Jun 15", "Group Stage", "F", "SWE", "TUN",  1),
    ("Jun 16", "Group Stage", "H", "ESP", "CPV",  1),
    ("Jun 16", "Group Stage", "G", "BEL", "EGY",  1),
    ("Jun 16", "Group Stage", "H", "KSA", "URU",  1),
    ("Jun 16", "Group Stage", "G", "IRN", "NZL",  1),
    ("Jun 17", "Group Stage", "I", "FRA", "SEN",  1),
    ("Jun 17", "Group Stage", "I", "IRQ", "NOR",  1),
    ("Jun 17", "Group Stage", "J", "ARG", "ALG",  1),
    ("Jun 17", "Group Stage", "J", "AUT", "JOR",  1),
    ("Jun 18", "Group Stage", "K", "POR", "COD",  1),
    ("Jun 18", "Group Stage", "L", "ENG", "CRO",  1),
    ("Jun 18", "Group Stage", "L", "GHA", "PAN",  1),
    ("Jun 18", "Group Stage", "K", "UZB", "COL",  1),
    # ── Matchday 2 ─────────────────────────────────────────────────────
    ("Jun 19", "Group Stage", "A", "CZE", "RSA",  2),
    ("Jun 19", "Group Stage", "B", "SUI", "BIH",  2),
    ("Jun 19", "Group Stage", "B", "CAN", "QAT",  2),
    ("Jun 19", "Group Stage", "A", "MEX", "KOR",  2),
    ("Jun 20", "Group Stage", "D", "USA", "AUS",  2),
    ("Jun 20", "Group Stage", "C", "SCO", "MAR",  2),
    ("Jun 20", "Group Stage", "C", "BRA", "HAI",  2),
    ("Jun 20", "Group Stage", "D", "TUR", "PAR",  2),
    ("Jun 21", "Group Stage", "F", "NED", "SWE",  2),
    ("Jun 21", "Group Stage", "E", "GER", "CIV",  2),
    ("Jun 21", "Group Stage", "E", "ECU", "CUW",  2),
    ("Jun 21", "Group Stage", "F", "TUN", "JPN",  2),
    ("Jun 22", "Group Stage", "H", "ESP", "KSA",  2),
    ("Jun 22", "Group Stage", "G", "BEL", "IRN",  2),
    ("Jun 22", "Group Stage", "H", "URU", "CPV",  2),
    ("Jun 22", "Group Stage", "G", "NZL", "EGY",  2),
    ("Jun 23", "Group Stage", "J", "ARG", "AUT",  2),
    ("Jun 23", "Group Stage", "I", "FRA", "IRQ",  2),
    ("Jun 23", "Group Stage", "I", "NOR", "SEN",  2),
    ("Jun 23", "Group Stage", "J", "JOR", "ALG",  2),
    ("Jun 24", "Group Stage", "K", "POR", "UZB",  2),
    ("Jun 24", "Group Stage", "L", "ENG", "GHA",  2),
    ("Jun 24", "Group Stage", "L", "PAN", "CRO",  2),
    ("Jun 24", "Group Stage", "K", "COL", "COD",  2),
    # ── Matchday 3 ─────────────────────────────────────────────────────
    ("Jun 25", "Group Stage", "B", "SUI", "CAN",  3),
    ("Jun 25", "Group Stage", "B", "BIH", "QAT",  3),
    ("Jun 25", "Group Stage", "C", "SCO", "BRA",  3),
    ("Jun 25", "Group Stage", "C", "MAR", "HAI",  3),
    ("Jun 25", "Group Stage", "A", "CZE", "MEX",  3),
    ("Jun 25", "Group Stage", "A", "RSA", "KOR",  3),
    ("Jun 26", "Group Stage", "E", "CUW", "CIV",  3),
    ("Jun 26", "Group Stage", "E", "ECU", "GER",  3),
    ("Jun 26", "Group Stage", "F", "JPN", "SWE",  3),
    ("Jun 26", "Group Stage", "F", "TUN", "NED",  3),
    ("Jun 26", "Group Stage", "D", "TUR", "USA",  3),
    ("Jun 26", "Group Stage", "D", "PAR", "AUS",  3),
    ("Jun 27", "Group Stage", "I", "NOR", "FRA",  3),
    ("Jun 27", "Group Stage", "I", "SEN", "IRQ",  3),
    ("Jun 27", "Group Stage", "H", "URU", "ESP",  3),
    ("Jun 27", "Group Stage", "H", "CPV", "KSA",  3),
    ("Jun 27", "Group Stage", "G", "NZL", "BEL",  3),
    ("Jun 27", "Group Stage", "G", "EGY", "IRN",  3),
    ("Jun 28", "Group Stage", "L", "PAN", "ENG",  3),
    ("Jun 28", "Group Stage", "L", "CRO", "GHA",  3),
    ("Jun 28", "Group Stage", "K", "COL", "POR",  3),
    ("Jun 28", "Group Stage", "K", "COD", "UZB",  3),
    ("Jun 28", "Group Stage", "J", "JOR", "ARG",  3),
    ("Jun 28", "Group Stage", "J", "ALG", "AUT",  3),
]

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


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class Fixture:
    match_num: int
    date: str
    round: str          # "Group Stage" | "Round of 32" | etc.
    group: str          # "A"–"L" for group stage, "" for knockout
    home: str
    away: str
    matchday: int       # 1/2/3 for group; 0 for knockout
    venue: str = ""     # Match host city
    # Result (filled after simulation)
    home_score: int = -1
    away_score: int = -1
    went_to_et: bool = False
    went_to_pens: bool = False
    pen_home: int = 0
    pen_away: int = 0
    winner: str = ""
    home_win_pct: float = 0.0
    draw_pct: float = 0.0
    away_win_pct: float = 0.0
    home_xg: float = 0.0
    away_xg: float = 0.0
    reasoning: list[str] = field(default_factory=list)

    @property
    def played(self) -> bool:
        return self.home_score >= 0

    def result_str(self) -> str:
        if not self.played:
            return "TBD"
        s = f"{self.home_score}–{self.away_score}"
        if self.went_to_pens:
            s += f" (pens {self.pen_home}–{self.pen_away})"
        elif self.went_to_et:
            s += " (AET)"
        return s


@dataclass
class TeamStanding:
    team: str
    group: str
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    @property
    def pts(self) -> int:
        return self.won * 3 + self.drawn


@dataclass
class TournamentResult:
    fixtures: list[Fixture] = field(default_factory=list)
    standings: dict[str, dict[str, TeamStanding]] = field(default_factory=dict)
    qualified: dict[str, str] = field(default_factory=dict)   # team -> "Winner"/"Runner-up"/"3rd"/"Eliminated"
    knockout_rounds: list[str] = field(default_factory=list)
    champion: str = ""
    runner_up: str = ""
    stats: dict = field(default_factory=dict)


# ── Fixture loader ─────────────────────────────────────────────────────────

def load_fixtures() -> list[Fixture]:
    """Build the ordered list of group-stage fixtures from the hard schedule."""
    GROUP_CITIES = {
        "A": "New York", "B": "Boston", "C": "Miami", "D": "Dallas", "E": "Houston", "F": "Kansas City",
        "G": "Toronto", "H": "Vancouver", "I": "Mexico City", "J": "Guadalajara", "K": "Los Angeles", "L": "San Francisco"
    }
    fixtures: list[Fixture] = []
    for i, (date, rnd, group, hc, ac, md) in enumerate(_RAW_FIXTURES, 1):
        fixtures.append(Fixture(
            match_num=i,
            date=date,
            round=rnd,
            group=group,
            home=TEAM_CODES[hc],
            away=TEAM_CODES[ac],
            matchday=md,
            venue=GROUP_CITIES.get(group, ""),
        ))
    return fixtures


# ── Standings helpers ──────────────────────────────────────────────────────

def build_empty_standings() -> dict[str, dict[str, TeamStanding]]:
    return {
        g: {t: TeamStanding(team=t, group=g) for t in teams}
        for g, teams in GROUPS.items()
    }

def update_standings(
    standings: dict[str, dict[str, TeamStanding]],
    fix: Fixture,
) -> None:
    """Apply a group-stage result to standings."""
    if fix.group not in standings:
        return
    hs, as_ = fix.home_score, fix.away_score
    h_st = standings[fix.group][fix.home]
    a_st = standings[fix.group][fix.away]
    h_st.played += 1; a_st.played += 1
    h_st.gf += hs;    h_st.ga += as_
    a_st.gf += as_;   a_st.ga += hs
    if hs > as_:
        h_st.won += 1; a_st.lost += 1
    elif hs < as_:
        a_st.won += 1; h_st.lost += 1
    else:
        h_st.drawn += 1; a_st.drawn += 1

def rank_group(group_standings: dict[str, TeamStanding]) -> list[TeamStanding]:
    return sorted(
        group_standings.values(),
        key=lambda s: (s.pts, s.gd, s.gf, s.team),
        reverse=True,
    )

def determine_qualifiers(
    standings: dict[str, dict[str, TeamStanding]]
) -> tuple[list[str], list[str], list[TeamStanding]]:
    """Return (winners, runners_up, all_thirds_sorted)."""
    winners, runners_up, thirds = [], [], []
    for g in sorted(standings.keys()):
        ranked = rank_group(standings[g])
        if len(ranked) >= 1:
            winners.append(ranked[0].team)
        if len(ranked) >= 2:
            runners_up.append(ranked[1].team)
        if len(ranked) >= 3:
            thirds.append(ranked[2])
    thirds_sorted = sorted(thirds, key=lambda s: (s.pts, s.gd, s.gf), reverse=True)
    return winners, runners_up, thirds_sorted


# ── Main predictor ─────────────────────────────────────────────────────────

class TournamentPredictor:
    """Drives a full fixture-by-fixture simulation of the World Cup."""

    def __init__(self, engine):
        self.engine = engine
        self.fatigue = defaultdict(float)
        self.travel_cities = {}

    # ── helpers ────────────────────────────────────────────────────────

    def _get_distance(self, city1: str, city2: str) -> float:
        CITY_COORDS = {
            "New York": (40.7128, -74.0060), "Boston": (42.3601, -71.0589), "Miami": (25.7617, -80.1918),
            "Dallas": (32.7767, -96.7970), "Houston": (29.7604, -95.3698), "Kansas City": (39.0997, -94.5786),
            "Toronto": (43.6532, -79.3832), "Vancouver": (49.2827, -123.1207), "Mexico City": (19.4326, -99.1332),
            "Guadalajara": (20.6597, -103.3496), "Monterrey": (25.6866, -100.3161), "Los Angeles": (34.0522, -118.2437),
            "San Francisco": (37.7749, -122.4194), "Atlanta": (33.7490, -84.3880), "Philadelphia": (39.9526, -75.1652),
            "Seattle": (47.6062, -122.3321)
        }
        if city1 not in CITY_COORDS or city2 not in CITY_COORDS:
            return 0.0
        lat1, lon1 = CITY_COORDS[city1]
        lat2, lon2 = CITY_COORDS[city2]
        import math
        return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 69.0

    def record_match_travel(self, team: str, city: str):
        if not hasattr(self, "travel_cities"):
            self.travel_cities = {}
        if not hasattr(self, "fatigue"):
            self.fatigue = defaultdict(float)

        if team in self.travel_cities:
            prev_city = self.travel_cities[team]
            dist = self._get_distance(prev_city, city)
            if dist > 0:
                self.fatigue[team] += dist / 5000.0
        self.travel_cities[team] = city

    def update_elo_recursive(self, home_t: str, away_t: str, home_score: int, away_score: int, is_ko: bool):
        h_rat = self.engine.ratings.get(home_t)
        a_rat = self.engine.ratings.get(away_t)
        if not h_rat or not a_rat:
            return
        from utils.engine import _new_elos
        outcome = 1.0 if home_score > away_score else (0.0 if away_score > home_score else 0.5)
        k = 60.0 if is_ko else 50.0
        new_h_elo, new_a_elo = _new_elos(h_rat.elo, a_rat.elo, outcome, k_factor=k, gd=abs(home_score - away_score))
        h_rat.elo = new_h_elo
        a_rat.elo = new_a_elo

    def update_momentum_recursive(self, home_t: str, away_t: str, home_score: int, away_score: int):
        h_rat = self.engine.ratings.get(home_t)
        a_rat = self.engine.ratings.get(away_t)
        if not h_rat or not a_rat:
            return
        h_rat.momentum_attack = 0.9 * h_rat.momentum_attack + 0.1 * home_score
        a_rat.momentum_attack = 0.9 * a_rat.momentum_attack + 0.1 * away_score
        h_rat.momentum_defense = 0.9 * h_rat.momentum_defense + 0.1 * away_score
        a_rat.momentum_defense = 0.9 * a_rat.momentum_defense + 0.1 * home_score

    def check_injuries_recursive(self, team: str):
        import random
        if random.random() < 0.02:
            rat = self.engine.ratings.get(team)
            if rat:
                rat.attack *= 0.90
                rat.defense *= 0.90

    def _build_reasoning(self, home: str, away: str) -> list[str]:
        h = self.engine._get_rating(home)
        a = self.engine._get_rating(away)
        reasons = []
        
        # Elo
        elo_diff = abs(h.elo - a.elo)
        if h.elo > a.elo:
            reasons.append(f"Higher Elo Rating ({h.elo:.0f} vs {a.elo:.0f})")
        else:
            reasons.append(f"Elo Rating difference favours {away} (+{elo_diff:.0f})")

        # Attack Strength
        if h.attack > a.attack:
            reasons.append(f"Superior attack rating ({h.attack:.2f} vs {a.attack:.2f})")
        elif a.attack > h.attack:
            reasons.append(f"Dangerous attack rating for {away} ({a.attack:.2f})")

        # Momentum
        if h.momentum_attack > a.momentum_attack:
            reasons.append(f"Better attacking momentum ({h.momentum_attack:.1f} vs {a.momentum_attack:.1f} goals)")
        elif a.momentum_attack > h.momentum_attack:
            reasons.append(f"Hotter scoring streak for {away} ({a.momentum_attack:.1f} goals)")

        # Host Boost or Slump
        hosts = {"United States", "Mexico", "Canada"}
        if home in hosts:
            reasons.append(f"Home Advantage: host boost for {home}")
        elif away in hosts:
            reasons.append(f"Home Advantage: host boost for {away}")

        if home == "Argentina":
            reasons.append("Slump risk: reigning champions underperform")
        elif away == "Argentina":
            reasons.append(f"Slump risk for {away}: reigning champions drag")

        return reasons[:4]

    def simulate_fixture_match(self, fix: Fixture, knockout: bool = False) -> Fixture:
        """Simulate one fixture and populate its result fields."""
        home_f = getattr(self, "fatigue", {}).get(fix.home, 0.0)
        away_f = getattr(self, "fatigue", {}).get(fix.away, 0.0)
        
        pred = self.engine.predict_match(fix.home, fix.away, venue=fix.venue, home_fatigue=home_f, away_fatigue=away_f)
        result = self.engine.simulate_match(
            fix.home, fix.away, knockout=knockout, venue=fix.venue, home_fatigue=home_f, away_fatigue=away_f
        )

        fix.home_score = result.home_score
        fix.away_score = result.away_score
        fix.went_to_et = result.went_to_et
        fix.went_to_pens = result.went_to_pens
        fix.pen_home = result.pen_home
        fix.pen_away = result.pen_away
        fix.home_win_pct = pred["home_win_pct"]
        fix.draw_pct = pred["draw_pct"]
        fix.away_win_pct = pred["away_win_pct"]
        fix.home_xg = pred["home_xg"]
        fix.away_xg = pred["away_xg"]
        fix.reasoning = self._build_reasoning(fix.home, fix.away)

        if knockout:
            fix.winner = result.winner or fix.home
        else:
            if fix.home_score > fix.away_score:
                fix.winner = fix.home
            elif fix.away_score > fix.home_score:
                fix.winner = fix.away
            else:
                fix.winner = "Draw"
                
        # Recursive updates during tournament
        self.update_elo_recursive(fix.home, fix.away, fix.home_score, fix.away_score, is_ko=knockout)
        self.update_momentum_recursive(fix.home, fix.away, fix.home_score, fix.away_score)
        self.check_injuries_recursive(fix.home)
        self.check_injuries_recursive(fix.away)
        
        return fix

    def simulate_group_phase(
        self,
        fixtures: list[Fixture],
        standings: dict[str, dict[str, TeamStanding]],
    ) -> list[Fixture]:
        """Simulate all group-stage fixtures in order with travel fatigue."""
        played = []
        for fix in fixtures:
            if fix.round != "Group Stage":
                continue
            self.record_match_travel(fix.home, fix.venue)
            self.record_match_travel(fix.away, fix.venue)
            self.simulate_fixture_match(fix, knockout=False)
            update_standings(standings, fix)
            played.append(fix)
        return played

    def generate_knockout_bracket(
        self,
        winners: list[str],
        runners_up: list[str],
        best_thirds: list[str],
        next_match_num: int,
    ) -> list[Fixture]:
        """
        Build Round of 32 fixtures using a deterministic seeding matching the official FIFA schema.
        """
        thirds = list(best_thirds)
        while len(thirds) < 8:
            thirds.append("Cape Verde")

        all_matchups = [
            (winners[0], runners_up[2], "New York"),          # W A vs RU C
            (winners[1], thirds[0], "Boston"),                # W B vs 3rd
            (winners[2], runners_up[0], "Miami"),             # W C vs RU A
            (winners[3], thirds[1], "Dallas"),                # W D vs 3rd
            (winners[4], runners_up[5], "Houston"),           # W E vs RU F
            (winners[5], runners_up[4], "Kansas City"),       # W F vs RU E
            (winners[6], thirds[2], "Toronto"),               # W G vs 3rd
            (winners[7], runners_up[9], "Vancouver"),         # W H vs RU J
            (winners[8], thirds[3], "Mexico City"),           # W I vs 3rd
            (winners[9], runners_up[7], "Guadalajara"),       # W J vs RU H
            (winners[10], runners_up[11], "Los Angeles"),     # W K vs RU L
            (winners[11], runners_up[10], "San Francisco"),   # W L vs RU K
            (runners_up[1], thirds[4], "Atlanta"),            # RU B vs 3rd
            (runners_up[3], thirds[5], "Philadelphia"),       # RU D vs 3rd
            (runners_up[6], thirds[6], "Seattle"),            # RU G vs 3rd
            (runners_up[8], thirds[7], "Monterrey"),          # RU I vs 3rd
        ]

        fixtures = []
        for i, (h, a, city) in enumerate(all_matchups):
            fixtures.append(Fixture(
                match_num=next_match_num + i,
                date="Jun 29+",
                round="Round of 32",
                group="",
                home=h,
                away=a,
                matchday=0,
                venue=city,
            ))
        return fixtures

    def simulate_knockout_phase(
        self,
        r32_fixtures: list[Fixture],
        next_match_num: int,
    ) -> tuple[list[Fixture], str, str]:
        """
        Simulate R32 → R16 → QF → SF → Bronze → Final.
        Returns (all_knockout_fixtures, champion, runner_up).
        """
        all_ko: list[Fixture] = []
        current_round_fixtures = r32_fixtures
        round_names = ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals"]
        next_round_names = ["Round of 16", "Quarter-Finals", "Semi-Finals", "Final"]

        # Knockout venues per round
        r16_cities = ["Seattle", "San Francisco", "Houston", "Dallas", "Atlanta", "Philadelphia", "New York", "Vancouver"]
        qf_cities = ["Boston", "Los Angeles", "Miami", "Kansas City"]
        sf_cities = ["Atlanta", "Dallas"]

        sf_losers: list[str] = []
        champion = ""
        runner_up = ""
        num = next_match_num

        for rnd_idx, rnd_name in enumerate(round_names):
            winners: list[str] = []
            for fix in current_round_fixtures:
                self.simulate_fixture_match(fix, knockout=True)
                all_ko.append(fix)
                w = fix.winner
                winners.append(w)
                if rnd_name == "Semi-Finals":
                    loser = fix.away if w == fix.home else fix.home
                    sf_losers.append(loser)

            if not winners:
                break

            if rnd_idx < len(next_round_names) - 1:
                # Build next round
                next_rnd_name = next_round_names[rnd_idx]
                next_fixtures = []
                cities = r16_cities if next_rnd_name == "Round of 16" else (qf_cities if next_rnd_name == "Quarter-Finals" else sf_cities)
                for i in range(0, len(winners), 2):
                    if i + 1 < len(winners):
                        next_fixtures.append(Fixture(
                            match_num=num,
                            date="TBD",
                            round=next_rnd_name,
                            group="",
                            home=winners[i],
                            away=winners[i + 1],
                            matchday=0,
                            venue=cities[i] if i < len(cities) else "Dallas",
                        ))
                        num += 1
                current_round_fixtures = next_fixtures
            else:
                # This was the semis — we have finalists
                if len(winners) >= 2:
                    # Bronze
                    if len(sf_losers) >= 2:
                        bronze = Fixture(num, "Jul 19", "Bronze Final", "", sf_losers[0], sf_losers[1], 0, venue="Miami")
                        self.simulate_fixture_match(bronze, knockout=True)
                        all_ko.append(bronze)
                        num += 1
                    # Final
                    final = Fixture(num, "Jul 20", "Final", "", winners[0], winners[1], 0, venue="New York")
                    self.simulate_fixture_match(final, knockout=True)
                    all_ko.append(final)
                    champion = final.winner
                    runner_up = final.away if champion == final.home else final.home

        return all_ko, champion, runner_up

    def _compute_stats(
        self,
        group_fixtures: list[Fixture],
        ko_fixtures: list[Fixture],
        standings: dict[str, dict[str, TeamStanding]],
    ) -> dict:
        all_fixtures = group_fixtures + ko_fixtures
        total_goals = sum(f.home_score + f.away_score for f in all_fixtures if f.played)
        
        # Highest scoring match
        hsm = max(all_fixtures, key=lambda f: f.home_score + f.away_score if f.played else 0)
        
        # Goals per team
        team_goals: dict[str, int] = defaultdict(int)
        team_conceded: dict[str, int] = defaultdict(int)
        for f in all_fixtures:
            if f.played:
                team_goals[f.home] += f.home_score
                team_goals[f.away] += f.away_score
                team_conceded[f.home] += f.away_score
                team_conceded[f.away] += f.home_score

        top_scorer_team = max(team_goals, key=team_goals.get) if team_goals else ""
        best_defense = min(team_conceded, key=team_conceded.get) if team_conceded else ""

        # Biggest upset: game where lower-elo team won
        biggest_upset = None
        biggest_upset_diff = 0
        for f in all_fixtures:
            if f.played and f.winner not in ("Draw", ""):
                h_elo = self.engine._get_rating(f.home).elo
                a_elo = self.engine._get_rating(f.away).elo
                if f.winner == f.away and h_elo > a_elo:
                    diff = h_elo - a_elo
                    if diff > biggest_upset_diff:
                        biggest_upset_diff = diff
                        biggest_upset = f
                elif f.winner == f.home and a_elo > h_elo:
                    diff = a_elo - h_elo
                    if diff > biggest_upset_diff:
                        biggest_upset_diff = diff
                        biggest_upset = f

        return {
            "total_goals": total_goals,
            "total_matches": len([f for f in all_fixtures if f.played]),
            "goals_per_match": round(total_goals / max(1, len([f for f in all_fixtures if f.played])), 2),
            "highest_scoring_match": f"{hsm.home} {hsm.home_score}–{hsm.away_score} {hsm.away}" if hsm.played else "",
            "top_scorer_team": top_scorer_team,
            "top_scorer_team_goals": team_goals.get(top_scorer_team, 0),
            "best_defense_team": best_defense,
            "best_defense_goals_conceded": team_conceded.get(best_defense, 0),
            "biggest_upset": f"{biggest_upset.winner} beat {biggest_upset.home if biggest_upset.winner==biggest_upset.away else biggest_upset.away} (Elo gap {biggest_upset_diff:.0f})" if biggest_upset else "None",
        }

    # ── Top-level ──────────────────────────────────────────────────────

    def predict_tournament(self) -> TournamentResult:
        """Run the full tournament simulation and return a TournamentResult."""
        from utils.engine import TeamRating
        
        # Back up original ratings of participating teams to avoid mutating global engine state
        participants = set()
        for teams in GROUPS.values():
            participants.update(teams)
        orig_ratings = {t: TeamRating(**vars(self.engine.ratings[t])) for t in participants if t in self.engine.ratings}
        
        self.fatigue = defaultdict(float)
        self.travel_cities = {}
        
        try:
            result = TournamentResult()
            result.standings = build_empty_standings()

            # 1. Load and simulate group stage
            group_fixtures = load_fixtures()
            self.simulate_group_phase(group_fixtures, result.standings)
            result.fixtures.extend(group_fixtures)

            # 2. Determine qualifiers
            winners, runners_up, thirds_sorted = determine_qualifiers(result.standings)
            best_thirds = [t.team for t in thirds_sorted[:8]]

            # Mark qualification status
            for t in winners:
                result.qualified[t] = "Group Winner ✅"
            for t in runners_up:
                result.qualified[t] = "Runner-Up ✅"
            for t in best_thirds:
                result.qualified[t] = "Best 3rd ✅"
            for g_standings in result.standings.values():
                for t, st in g_standings.items():
                    if t not in result.qualified:
                        result.qualified[t] = "Eliminated ❌"

            # 3. Build R32
            next_num = len(group_fixtures) + 1
            r32 = self.generate_knockout_bracket(winners, runners_up, best_thirds, next_num)

            # 4. Simulate knockouts
            all_ko, champion, runner_up = self.simulate_knockout_phase(
                r32, next_num + len(r32)
            )
            result.fixtures.extend(r32)
            result.fixtures.extend(all_ko)
            result.champion = champion
            result.runner_up = runner_up

            # 5. Stats
            result.stats = self._compute_stats(group_fixtures, r32 + all_ko, result.standings)

            # Order of knockout rounds for display
            result.knockout_rounds = ["Round of 32", "Round of 16", "Quarter-Finals",
                                       "Semi-Finals", "Bronze Final", "Final"]
            return result
        finally:
            for t, r in orig_ratings.items():
                self.engine.ratings[t] = r

    # ── Export ─────────────────────────────────────────────────────────

    @staticmethod
    def export_prediction_report(result: TournamentResult, fmt: str = "csv") -> bytes:
        """Export full prediction data as CSV, JSON, or a text summary."""
        rows = []
        for f in result.fixtures:
            if f.played:
                rows.append({
                    "match": f.match_num,
                    "date": f.date,
                    "round": f.round,
                    "group": f.group,
                    "home": f.home,
                    "away": f.away,
                    "home_score": f.home_score,
                    "away_score": f.away_score,
                    "winner": f.winner,
                    "aet": f.went_to_et,
                    "penalties": f.went_to_pens,
                    "pen_home": f.pen_home if f.went_to_pens else "",
                    "pen_away": f.pen_away if f.went_to_pens else "",
                    "home_win_pct": f.home_win_pct,
                    "draw_pct": f.draw_pct,
                    "away_win_pct": f.away_win_pct,
                })

        df = pd.DataFrame(rows)
        if fmt == "csv":
            return df.to_csv(index=False).encode()
        elif fmt == "json":
            return df.to_json(orient="records", indent=2).encode()
        else:  # text summary
            lines = [
                "FIFA WORLD CUP 2026 — AI PREDICTION REPORT",
                "=" * 50,
                f"Champion: {result.champion}",
                f"Runner-Up: {result.runner_up}",
                f"Total Goals: {result.stats.get('total_goals', 0)}",
                f"Goals/Match: {result.stats.get('goals_per_match', 0)}",
                "",
                "GROUP STANDINGS",
                "-" * 30,
            ]
            for g in sorted(result.standings.keys()):
                lines.append(f"\nGroup {g}")
                for st in rank_group(result.standings[g]):
                    lines.append(f"  {st.pts:2d}  {st.team}")
            lines += ["", "ALL RESULTS", "-" * 30]
            for f in result.fixtures:
                if f.played:
                    lines.append(f"  {f.round:15s}  {f.home:25s} {f.home_score}-{f.away_score}  {f.away}")
            return "\n".join(lines).encode()
