import sys
from pathlib import Path

# Add the project root to the python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.engine import FootballEngine

def main():
    engine = FootballEngine()
    engine.load_data()
    engine.compute_ratings()
    
    teams = ["France", "Spain", "Argentina", "Japan", "Belgium"]
    
    print(f"{'Team':<12} | {'Elo':<7} | {'Attack':<6} | {'Defense':<7} | {'Avg Scored':<10} | {'Avg Conced':<10} | {'Form':<5} | {'Momentum Att':<12}")
    print("-" * 90)
    for name in teams:
        t = engine._get_rating(name)
        print(f"{t.name:<12} | {t.elo:<7.1f} | {t.attack:<6.3f} | {t.defense:<7.3f} | {t.goals_scored_avg:<10.3f} | {t.goals_conceded_avg:<10.3f} | {t.recent_form:<5.2f} | {t.momentum_attack:<12.3f}")

if __name__ == "__main__":
    main()
