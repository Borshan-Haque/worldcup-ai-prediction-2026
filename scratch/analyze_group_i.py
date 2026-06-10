import sys
from pathlib import Path

# Add the project root to the python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.engine import FootballEngine

def main():
    engine = FootballEngine()
    engine.load_data()
    engine.compute_ratings()
    
    group_teams = ["France", "Senegal", "Iraq", "Norway"]
    print("Group I Match Predictions:")
    print("=" * 60)
    
    for i, h in enumerate(group_teams):
        for j, a in enumerate(group_teams):
            if i >= j:
                continue
            res = engine.predict_match(h, a)
            print(f"{h} vs {a}:")
            print(f"  Expected Goals: {h} {res['home_xg']} - {res['away_xg']} {a}")
            print(f"  Probabilities: Win {h}: {res['home_win_pct']}%, Draw: {res['draw_pct']}%, Win {a}: {res['away_win_pct']}%")
            print(f"  Most likely score: {res['predicted_home']}-{res['predicted_away']}")
            print("-" * 60)

    # Let's also compare France against some other top team in their group
    print("\nCompare with Spain's Group (Group H: Spain, Cape Verde, Saudi Arabia, Uruguay):")
    group_h = ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"]
    for i, h in enumerate(group_h):
        for j, a in enumerate(group_h):
            if i >= j:
                continue
            res = engine.predict_match(h, a)
            print(f"{h} vs {a}:")
            print(f"  Expected Goals: {h} {res['home_xg']} - {res['away_xg']} {a}")
            print(f"  Probabilities: Win {h}: {res['home_win_pct']}%, Draw: {res['draw_pct']}%, Win {a}: {res['away_win_pct']}%")
            print("-" * 60)

if __name__ == "__main__":
    main()
