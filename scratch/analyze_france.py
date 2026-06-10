import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Add the project root to the python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.engine import FootballEngine

def main():
    engine = FootballEngine()
    print("Loading data...")
    engine.load_data()
    print("Computing ratings...")
    engine.compute_ratings()
    
    print("\nFrance Ratings:")
    f_rating = engine._get_rating("France")
    print(f"Elo: {f_rating.elo:.2f}")
    print(f"Attack: {f_rating.attack:.3f}")
    print(f"Defense: {f_rating.defense:.3f}")
    print(f"Goals Scored Avg: {f_rating.goals_scored_avg:.3f}")
    print(f"Goals Conceded Avg: {f_rating.goals_conceded_avg:.3f}")
    print(f"Recent Form: {f_rating.recent_form:.3f}")
    print(f"Momentum Attack: {f_rating.momentum_attack:.3f}")
    print(f"Momentum Defense: {f_rating.momentum_defense:.3f}")
    
    print("\nRunning 5000 simulations...")
    n_sims = 5000
    
    group_stats = defaultdict(int)
    round_stats = defaultdict(int) # R32, R16, QF, SF, Bronze, Final, Champion
    
    for _ in range(n_sims):
        bracket = engine.simulate_tournament()
        
        # Check group stage
        qualified = False
        # Let's find France's group standings
        group_i = bracket["groups"]["I"]
        # France is in Group I
        qualified_teams = list(group_i.head(2)["team"])
        # Check if France is one of the top 2
        if "France" in qualified_teams:
            qualified = True
            group_stats["Top 2"] += 1
        else:
            # Check if France is in 3rd place and qualified as wild card
            third_place_team = group_i.iloc[2]["team"]
            # To see if they made it to R32, let's look at bracket["Round of 32"]
            # But the R32 results contain match objects. Let's see who is in R32
            r32_teams = []
            for match in bracket.get("Round of 32", []):
                r32_teams.extend([match.home, match.away])
            if "France" in r32_teams:
                qualified = True
                group_stats["Wild Card"] += 1
            else:
                group_stats["Eliminated in Groups"] += 1
                
        # Knockout rounds
        current_teams = []
        for rnd in ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals", "Final"]:
            matches = bracket.get(rnd, [])
            round_teams = []
            for m in matches:
                round_teams.extend([m.home, m.away])
            if "France" in round_teams:
                round_stats[rnd] += 1
                
        if bracket.get("Champion") == "France":
            round_stats["Champion"] += 1
            
    print("\nSimulation Results for France:")
    print(f"Group Stage:")
    print(f"  Top 2 qualification: {group_stats['Top 2'] / n_sims * 100:.2f}%")
    print(f"  Wild Card qualification: {group_stats['Wild Card'] / n_sims * 100:.2f}%")
    print(f"  Eliminated in Group Stage: {group_stats['Eliminated in Groups'] / n_sims * 100:.2f}%")
    
    print(f"\nKnockout Rounds:")
    for rnd in ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals", "Final", "Champion"]:
        pct = round_stats[rnd] / n_sims * 100
        print(f"  Reached {rnd}: {pct:.2f}%")

if __name__ == "__main__":
    main()
