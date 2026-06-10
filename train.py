"""
train.py — Build and cache team ratings from historical data.
Run: python train.py
"""

import logging
from utils.engine import FootballEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

if __name__ == "__main__":
    engine = FootballEngine()
    engine.load_data()
    engine.compute_ratings()
    print("\n✅  Training complete. Team ratings saved to data/team_ratings.csv")
    print("    Now run: streamlit run dashboard/app.py")
