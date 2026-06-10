"""
WorldCupAI — Tournament Predictor Engine
Parses the official fixture schedule and simulates the full tournament match-by-match.
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

# ---------------------------------------------------------------------------
# Fixture data — parsed directly from the PDF schedule
# ---------------------------------------------------------------------------

# Code → Full team name
TEAM_CODES: dict[str, str] = {
    "MEX": "Mexico", "RSA": "South Africa", "KOR": "South Korea", "CZE": "Czech Republic",
    "CAN": "Canada", "BIH": "Bosnia and Herzegovina", "QAT": "Qatar", "SUI": "Switzerland",
    "BRA": "Brazil", "MAR": "Morocco", "HAI": "Haiti", "SCO": "Scotland",
    "USA": "United States", "PAR": "Paraguay", "AUS": "Australia", "TUR": "Turkey",
    "GER": "Germany", "CUW": "Curaçao", "CIV": "Ivory Coast", "ECU": "Ecuador",
    "NED": "Netherlands", "JPN": "Japan", "SWE": "Sweden", "TUN": "Tunisia",
    "BEL": "Belgium", "EGY": "Egypt", "IRN": "Iran", "NZL": "New Zealand",
    "ESP": "Spain", "CPV": "Cape Verde", "KSA": "Saudi Arabia", "URU": "Uruguay",
    "FRA": "France", "SEN": "Senegal", "IRQ": "Iraq", "NOR": "Norway",
    "ARG": "Argentina", "ALG": "Algeria", "AUT": "Austria", "JOR": "Jordan",
    "POR": "Portugal", "COD": "DR Congo", "UZB": "Uzbekistan", "COL": "Colombia",
    "ENG": "England", "CRO": "Croatia", "GHA": "Ghana", "PAN": "Panama",
}

# Official group stage fixtures from the PDF, in chronological order
# Format: (date, group, home_code, away_code, matchday)
RAW_FIXTURES = [
    # Matchday 1
    ("2026-06-12", "A", "MEX", "RSA", 1),
    ("2026-06-12", "A", "KOR", "CZE", 1),
    ("2026-06-13", "B", "CAN", "BIH", 1),
    ("2026-06-13", "D", "USA", "PAR", 1),
    ("2026-06-14", "B", "QAT", "SUI", 1),
    ("2026-06-14", "C", "BRA", "MAR", 1),
    ("2026-06-14", "C", "HAI", "SCO", 1),
    ("2026-06-14", "D", "AUS", "TUR", 1),
    ("2026-06-15", "E", "GER", "CUW", 1),
    ("2026-06-15", "F", "NED", "JPN", 1),
    ("2026-06-15", "E", "CIV", "ECU", 1),
    ("2026-06-15", "F", "SWE", "TUN", 1),
    ("2026-06-16", "H", "ESP", "CPV", 1),
    ("2026-06-16", "G", "BEL", "EGY", 1),
    ("2026-06-16", "H", "KSA", "URU", 1),
    ("2026-06-16", "G", "IRN", "NZL", 1),
    ("2026-06-17", "I", "FRA", "SEN", 1),
    ("2026-06-17", "I", "IRQ", "NOR", 1),
    ("2026-06-17", "J", "ARG", "ALG", 1),
    ("2026-06-17", "J", "AUT", "JOR", 1),
    ("2026-06-18", "K", "POR", "COD", 1),
    ("2026-06-18", "L", "ENG", "CRO", 1),
    ("2026-06-18", "L", "GHA", "PAN", 1),
    ("2026-06-18", "K", "UZB", "COL", 1),
    # Matchday 2
    ("2026-06-19", "A", "CZE", "RSA", 2),
    ("2026-06-19", "B", "SUI", "BIH", 2),
    ("2026-06-19", "B", "CAN", "QAT", 2),
    ("2026-06-19", "A", "MEX", "KOR", 2),
    ("2026-06-20", "D", "USA", "AUS", 2),
    ("2026-06-20", "C", "SCO", "MAR", 2),
    ("2026-06-20", "C", "BRA", "HAI", 2),
    ("2026-06-20", "D", "TUR", "PAR", 2),
    ("2026-06-21", "F", "NED", "SWE", 2),
    ("2026-06-21", "E", "GER", "CIV", 2),
    ("2026-06-21", "E", "ECU", "CUW", 2),
    ("2026-06-21", "F", "TUN", "JPN", 2),
    ("2026-06-22", "H", "ESP", "KSA", 2),
    ("2026-06-22", "G", "BEL", "IRN", 2),
    ("2026-06-22", "H", "URU", "CPV", 2),
    ("2026-06-22", "G", "NZL", "EGY", 2),
    ("2026-06-23", "J", "ARG", "AUT", 2),
    ("2026-06-23", "I", "FRA", "IRQ", 2),
    ("2026-06-23", "I", "NOR", "SEN", 2),
    ("2026-06-23", "J", "JOR", "ALG", 2),
    ("2026-06-24", "K", "POR", "UZB", 2),
    ("2026-06-24", "L", "ENG", "GHA", 2),
    ("2026-06-24", "L", "PAN", "CRO", 2),
    ("2026-06-24", "K", "COL", "COD", 2),
    # Matchday 3
    ("2026-06-25", "B", "SUI", "CAN", 3),
    ("2026-06-25", "B", "BIH", "QAT", 3),
    ("2026-06-25", "C", "SCO", "BRA", 3),
    ("2026-06-25", "C", "MAR", "HAI", 3),
    ("2026-06-25", "A", "CZE", "MEX", 3),
    ("2026-06-25", "A", "RSA", "KOR", 3),
    ("2026-06-26", "E", "CUW", "CIV", 3),
    ("2026-06-26", "E", "ECU", "GER", 3),
    ("2026-06-26", "F", "JPN", "SWE", 3),
    ("2026-06-26", "F", "TUN", "NED", 3),
    ("2026-06-26", "D", "TUR", "USA", 3),
    ("2026-06-26", "D", "PAR", "AUS", 3),
    ("2026-06-27", "I", "NOR", "FRA", 3),
    ("2026-06-27", "I", "SEN", "IRQ", 3),
    ("2026-06-27", "H", "URU", "ESP", 3),
    ("2026-06-27", "H", "CPV", "KSA", 3),
    ("2026-06-27", "G", "NZL", "BEL", 3),
    ("2026-06-27", "G", "EGY", "IRN", 3),
    ("2026-06-28", "L", "PAN", "ENG", 3),
    ("2026-06-28", "L", "CRO", "GHA", 3),
    ("2026-06-28", "K", "COL", "POR", 3),
    ("2026-06-28", "K", "COD", "UZB", 3),
    ("2026-06-28", "J", "JOR", "ARG", 3),
    ("2026-06-28", "J", "ALG", "AUT", 3),
]

@dataclass
class Fixture:
    date: str
    group: str
    home: str
    away: str
    matchday: int
    home_score: int = -1
    away_score: int = -1
    home_xg: float = 0.0
    away_xg: float = 0.0
    home_win_pct: float = 0.0
    draw_pct: float = 0.0
    away_win_pct: float = 0.0
    went_to_et: bool = False
    went_to_pens: bool = False
    pen_home: int = 0
    pen_away: int = 0
    reasons: list[str] = field(default_factory=list)
    played: bool = False

    @property
    def winner(self) -> Optional[str]:
        if not self.played:
            return None
        if self.went_to_pens:
            return self.home if self.pen_home > self.pen_away else self.away
        if self.home_score > self.away_score:
            return self.home
        if self.away_score > self.home_score:
            return self.away
        return None

    @property
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
class KnockoutFixture:
    round_name: str
    match_num: int
    home: str = "TBD"
    away: str = "TBD"
    home_score: int = -1
    away_score: int = -1
    home_xg: float = 0.0
    away_xg: float = 0.0
    home_win_pct: float = 0.0
    away_win_pct: float = 0.0
    went_to_et: bool = False
    went_to_pens: bool = False
    pen_home: int = 0
    pen_away: int = 0
    played: bool = False

    @property
    def winner(self) -> Optional[str]:
        if not self.played:
            return None
        if self.went_to_pens:
            return self.home if self.pen_home > self.pen_away else self.away
        return self.home if self.home_score > self.away_score else self.away

    @property
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
class GroupStanding:
    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def pts(self) -> int:
        return self.wins * 3 + self.draws

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    def to_dict(self) -> dict:
        return {
            "team": self.team, "P": self.played,
            "W": self.wins, "D": self.draws, "L": self.losses,
            "GF": self.gf, "GA": self.ga, "GD": self.gd, "PTS": self.pts,
        }


@dataclass
class TournamentState:
    fixtures: list[Fixture] = field(default_factory=list)
    ko_fixtures: list[KnockoutFixture] = field(default_factory=list)
    standings: dict[str, dict[str, GroupStanding]] = field(default_factory=dict)
    qualified: dict[str, str] = field(default_factory=dict)   # team → "qualified"/"eliminated"
    champion: str = ""
    runner_up: str = ""
    third_place: str = ""
    stats: dict = field(default_factory=dict)
    complete: bool = False


# ---------------------------------------------------------------------------
# Tournament Predictor
# ---------------------------------------------------------------------------

class TournamentPredictor:
    """Simulates the entire FIFA World Cup 2026 fixture by fixture."""

    GROUPS = {
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

    def __init__(self, engine):
        self.engine = engine

    def _build_fixture_list(self) -> list[Fixture]:
        fixtures = []
        for date, group, home_code, away_code, matchday in RAW_FIXTURES:
            home = TEAM_CODES[home_code]
            away = TEAM_CODES[away_code]
            fixtures.append(Fixture(
                date=date, group=group, home=home, away=away, matchday=matchday
            ))
        return fixtures

    def _init_standings(self) -> dict[str, dict[str, GroupStanding]]:
        return {
            g: {t: GroupStanding(team=t) for t in teams}
            for g, teams in self.GROUPS.items()
        }

    def _build_reasons(self, home: str, away: str) -> list[str]:
        """Generate human-readable reasoning for the prediction."""
        hr = self.engine._get_rating(home)
        ar = self.engine._get_rating(away)
        reasons = []
        
        if hr.elo > ar.elo:
            reasons.append(f"✓ Higher Elo Rating ({hr.elo:.0f} vs {ar.elo:.0f})")
        else:
            reasons.append(f"✗ Lower Elo Rating ({hr.elo:.0f} vs {ar.elo:.0f})")
            
        if hr.attack > ar.attack:
            reasons.append(f"✓ Stronger Attack Rating ({hr.attack:.2f} vs {ar.attack:.2f})")
        else:
            reasons.append(f"✗ Weaker Attack Rating ({hr.attack:.2f} vs {ar.attack:.2f})")

        if hr.momentum_attack > ar.momentum_attack:
            reasons.append(f"✓ Higher Scoring Momentum ({hr.momentum_attack:.1f} vs {ar.momentum_attack:.1f})")
        else:
            reasons.append(f"✗ Lower Scoring Momentum ({hr.momentum_attack:.1f} vs {ar.momentum_attack:.1f})")

        hosts = {"United States", "Mexico", "Canada"}
        if home in hosts:
            reasons.append(f"✓ Host Boost: playing in {home}")
        elif away in hosts:
            reasons.append(f"✗ Opponent Host Boost: playing in {away}")

        if home == "Argentina":
            reasons.append("⚠ Winner's Slump factor for Argentina")
        elif away == "Argentina":
            reasons.append("⚠ Opponent is Reigning Champion")
            
        return reasons[:4]

    def simulate_fixture_match(self, fix: Fixture) -> Fixture:
        """Simulate a single group stage fixture in place."""
        pred = self.engine.predict_match(fix.home, fix.away)
        result = self.engine.simulate_match(fix.home, fix.away, knockout=False)

        fix.home_score = result.home_score
        fix.away_score = result.away_score
        fix.home_xg = pred["home_xg"]
        fix.away_xg = pred["away_xg"]
        fix.home_win_pct = pred["home_win_pct"]
        fix.draw_pct = pred["draw_pct"]
        fix.away_win_pct = pred["away_win_pct"]
        fix.reasons = self._build_reasons(fix.home, fix.away)
        fix.played = True
        return fix

    def _update_standing(self, standings: dict, group: str, fix: Fixture) -> None:
        h = standings[group][fix.home]
        a = standings[group][fix.away]
        h.played += 1; a.played += 1
        h.gf += fix.home_score; h.ga += fix.away_score
        a.gf += fix.away_score; a.ga += fix.home_score
        if fix.home_score > fix.away_score:
            h.wins += 1; a.losses += 1
        elif fix.home_score < fix.away_score:
            a.wins += 1; h.losses += 1
        else:
            h.draws += 1; a.draws += 1

    def _rank_group(self, standings: dict[str, GroupStanding]) -> list[GroupStanding]:
        return sorted(
            standings.values(),
            key=lambda s: (s.pts, s.gd, s.gf),
            reverse=True,
        )

    def simulate_group_phase(
        self,
        fixtures: list[Fixture],
        standings: dict[str, dict[str, GroupStanding]],
    ) -> None:
        """Simulate all group fixtures and update standings in place."""
        for fix in fixtures:
            self.simulate_fixture_match(fix)
            self._update_standing(standings, fix.group, fix)

    def _select_best_third_place(
        self, standings: dict[str, dict[str, GroupStanding]]
    ) -> list[str]:
        """Return the 8 best third-placed teams."""
        thirds = []
        for g, s in standings.items():
            ranked = self._rank_group(s)
            if len(ranked) >= 3:
                thirds.append(ranked[2])
        thirds.sort(key=lambda s: (s.pts, s.gd, s.gf), reverse=True)
        return [s.team for s in thirds[:8]]

    def generate_knockout_bracket(
        self, standings: dict[str, dict[str, GroupStanding]]
    ) -> list[KnockoutFixture]:
        """
        Build the Round of 32 using FIFA 2026 seeding rules.
        12 group winners + 12 runners-up + 8 best 3rd-place = 32 teams.
        """
        winners = {}
        runners = {}
        for g, s in standings.items():
            ranked = self._rank_group(s)
            winners[g] = ranked[0].team
            runners[g] = ranked[1].team

        thirds = self._select_best_third_place(standings)

        # FIFA 2026 R32 draw — simplified seeded pairing
        # W(A) vs best-3rd, W(B) vs R(F), W(C) vs R(E), W(D) vs best-3rd, etc.
        # We use a balanced pairing: winners face runners from opposite half
        group_order = list("ABCDEFGHIJKL")
        w_list = [winners[g] for g in group_order]
        r_list = [runners[g] for g in group_order]

        # Pair: each winner vs a runner-up from opposite side + 8 third-place fill-ins
        # For clarity: 16 R32 matches using w vs r pairings + third-place teams
        # Standard: W(A) v 3rd, W(B) v R(A), W(C) v R(D) ... simplified balanced draw
        participants_home = w_list[:8] + w_list[8:] 
        participants_away = r_list[6:] + r_list[:6]
        # Replace last 8 away slots with best third-place to fill 32-team bracket
        # Actually FIFA 2026: 12 winners, 12 R-U, 8 best-3rd → 32 total
        # Simple approach: interleave winners with runners + thirds
        all_32 = w_list + r_list[:8] + thirds[:8] + r_list[8:]
        import random as _r
        _r.shuffle(all_32)

        ko_fixtures = []
        for i in range(0, 32, 2):
            ko_fixtures.append(KnockoutFixture(
                round_name="Round of 32",
                match_num=i // 2 + 1,
                home=all_32[i],
                away=all_32[i + 1],
            ))
        return ko_fixtures

    def simulate_knockout_phase(
        self, ko_fixtures: list[KnockoutFixture]
    ) -> tuple[list[KnockoutFixture], str, str, str]:
        """
        Simulate all knockout rounds from R32 to Final.
        Returns (all ko fixtures, champion, runner_up, third_place).
        """
        all_ko: list[KnockoutFixture] = []
        current_round = ko_fixtures
        round_names = ["Round of 16", "Quarter-Finals", "Semi-Finals", "Final"]
        ri = 0

        # simulate R32
        for fix in current_round:
            self._simulate_ko_fixture(fix)
        all_ko.extend(current_round)
        winners = [f.winner for f in current_round]

        while len(winners) > 1:
            rname = round_names[min(ri, len(round_names) - 1)]
            next_round = []
            for i in range(0, len(winners), 2):
                if i + 1 < len(winners):
                    kf = KnockoutFixture(
                        round_name=rname,
                        match_num=i // 2 + 1,
                        home=winners[i],
                        away=winners[i + 1],
                    )
                    self._simulate_ko_fixture(kf)
                    next_round.append(kf)
            all_ko.extend(next_round)

            if rname == "Semi-Finals":
                # Bronze match
                losers = []
                for f in next_round:
                    loser = f.away if f.winner == f.home else f.home
                    losers.append(loser)
                if len(losers) >= 2:
                    bronze = KnockoutFixture(
                        round_name="Bronze Final",
                        match_num=1,
                        home=losers[0],
                        away=losers[1],
                    )
                    self._simulate_ko_fixture(bronze)
                    all_ko.append(bronze)
                    third_place = bronze.winner or losers[0]
                else:
                    third_place = ""
            
            winners = [f.winner for f in next_round]
            ri += 1

        champion = winners[0] if winners else ""
        # Find runner-up from final
        runner_up = ""
        final_matches = [f for f in all_ko if f.round_name == "Final"]
        if final_matches:
            f = final_matches[0]
            runner_up = f.away if f.winner == f.home else f.home
        
        third = ""
        bronze_matches = [f for f in all_ko if f.round_name == "Bronze Final"]
        if bronze_matches:
            bf = bronze_matches[0]
            third = bf.winner or bf.home

        return all_ko, champion, runner_up, third

    def _simulate_ko_fixture(self, kf: KnockoutFixture) -> None:
        pred = self.engine.predict_match(kf.home, kf.away)
        result = self.engine.simulate_match(kf.home, kf.away, knockout=True)
        kf.home_score = result.home_score
        kf.away_score = result.away_score
        kf.home_xg = pred["home_xg"]
        kf.away_xg = pred["away_xg"]
        kf.home_win_pct = pred["home_win_pct"]
        kf.away_win_pct = pred["away_win_pct"]
        kf.went_to_et = result.went_to_et
        kf.went_to_pens = result.went_to_pens
        kf.pen_home = result.pen_home
        kf.pen_away = result.pen_away
        kf.played = True

    def predict_tournament(self) -> TournamentState:
        """Run the full tournament prediction end-to-end."""
        state = TournamentState()
        state.fixtures = self._build_fixture_list()
        state.standings = self._init_standings()

        # Group phase
        self.simulate_group_phase(state.fixtures, state.standings)

        # Determine qualification
        for g, s in state.standings.items():
            ranked = self._rank_group(s)
            for i, standing in enumerate(ranked):
                state.qualified[standing.team] = "qualified" if i < 2 else "eliminated"
        thirds = self._select_best_third_place(state.standings)
        for t in thirds:
            state.qualified[t] = "qualified"

        # Knockout phase
        r32 = self.generate_knockout_bracket(state.standings)
        all_ko, champion, runner_up, third = self.simulate_knockout_phase(r32)
        state.ko_fixtures = all_ko
        state.champion = champion
        state.runner_up = runner_up
        state.third_place = third

        # Stats
        state.stats = self._compute_stats(state)
        state.complete = True
        return state

    def _compute_stats(self, state: TournamentState) -> dict:
        total_goals = sum(f.home_score + f.away_score for f in state.fixtures if f.played)
        total_goals += sum(
            f.home_score + f.away_score for f in state.ko_fixtures if f.played
        )

        all_played = [f for f in state.fixtures if f.played]
        if all_played:
            highest = max(all_played, key=lambda f: f.home_score + f.away_score)
            high_str = f"{highest.home} {highest.home_score}–{highest.away_score} {highest.away}"
        else:
            high_str = "N/A"

        # Goals per team across group stage
        team_gf: dict[str, int] = defaultdict(int)
        team_ga: dict[str, int] = defaultdict(int)
        for f in all_played:
            team_gf[f.home] += f.home_score
            team_gf[f.away] += f.away_score
            team_ga[f.home] += f.away_score
            team_ga[f.away] += f.home_score

        top_scorer_team = max(team_gf, key=team_gf.get) if team_gf else "N/A"
        best_defense = min(team_ga, key=team_ga.get) if team_ga else "N/A"

        # Biggest upset: match where lower-elo team won
        biggest_upset = "N/A"
        max_elo_diff = 0
        for f in all_played:
            if f.winner:
                hr = self.engine._get_rating(f.home)
                ar = self.engine._get_rating(f.away)
                if f.winner == f.away and (hr.elo - ar.elo) > max_elo_diff:
                    max_elo_diff = hr.elo - ar.elo
                    biggest_upset = f"{f.away} beat {f.home} ({max_elo_diff:.0f} Elo gap)"

        return {
            "total_goals": total_goals,
            "total_matches": len(all_played) + len([f for f in state.ko_fixtures if f.played]),
            "highest_scoring": high_str,
            "biggest_upset": biggest_upset,
            "top_attack": top_scorer_team,
            "top_attack_goals": team_gf.get(top_scorer_team, 0),
            "best_defense": best_defense,
            "best_defense_ga": team_ga.get(best_defense, 0),
        }

    def export_to_json(self, state: TournamentState) -> str:
        """Export full tournament state to JSON string."""
        data = {
            "champion": state.champion,
            "runner_up": state.runner_up,
            "third_place": state.third_place,
            "stats": state.stats,
            "group_fixtures": [
                {
                    "date": f.date, "group": f.group,
                    "home": f.home, "away": f.away, "matchday": f.matchday,
                    "score": f.result_str,
                    "home_win_pct": f.home_win_pct, "away_win_pct": f.away_win_pct,
                }
                for f in state.fixtures
            ],
            "knockout_fixtures": [
                {
                    "round": f.round_name, "match": f.match_num,
                    "home": f.home, "away": f.away,
                    "score": f.result_str,
                }
                for f in state.ko_fixtures
            ],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def export_to_csv(self, state: TournamentState) -> str:
        rows = []
        for f in state.fixtures:
            rows.append({
                "stage": "Group Stage", "round": f"Group {f.group} MD{f.matchday}",
                "home": f.home, "away": f.away, "score": f.result_str,
                "home_win_pct": f.home_win_pct, "away_win_pct": f.away_win_pct,
            })
        for f in state.ko_fixtures:
            rows.append({
                "stage": "Knockout", "round": f.round_name,
                "home": f.home, "away": f.away, "score": f.result_str,
                "home_win_pct": f.home_win_pct, "away_win_pct": f.away_win_pct,
            })
        return pd.DataFrame(rows).to_csv(index=False)
