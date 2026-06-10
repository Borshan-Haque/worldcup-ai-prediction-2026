import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Add the project root to the python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.engine import FootballEngine

def main():
    engine = FootballEngine()
    engine.load_data()
    engine.compute_ratings()
    
    n_sims = 200
    group_stats = defaultdict(int)
    round_stats = defaultdict(int)
    
    for _ in range(n_sims):
        bracket = engine.simulate_tournament()
        
        # Check group stage
        qualified = False
        group_i = bracket["groups"]["I"]
        qualified_teams = list(group_i.head(2)["team"])
        if "France" in qualified_teams:
            qualified = True
            group_stats["Top 2"] += 1
        else:
            r32_teams = []
            for match in bracket.get("Round of 32", []):
                r32_teams.extend([match.home, match.away])
            if "France" in r32_teams:
                qualified = True
                group_stats["Wild Card"] += 1
            else:
                group_stats["Eliminated in Groups"] += 1
                
        for rnd in ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals", "Final"]:
            matches = bracket.get(rnd, [])
            round_teams = []
            for m in matches:
                round_teams.extend([m.home, m.away])
            if "France" in round_teams:
                round_stats[rnd] += 1
                
        if bracket.get("Champion") == "France":
            round_stats["Champion"] += 1
            
    print("\nFrance Group Stage:")
    print(f"Top 2: {group_stats['Top 2'] / n_sims * 100:.1f}%")
    print(f"Wild Card: {group_stats['Wild Card'] / n_sims * 100:.1f}%")
    print(f"Eliminated: {group_stats['Eliminated in Groups'] / n_sims * 100:.1f}%")
    
    print("\nFrance Knockout Rounds:")
    for rnd in ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals", "Final", "Champion"]:
        print(f"{rnd}: {round_stats[rnd] / n_sims * 100:.1f}%")

if __name__ == "__main__":
    main()
