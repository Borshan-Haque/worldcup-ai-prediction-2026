import time
from utils.engine import FootballEngine

engine = FootballEngine()
engine.load_ratings()

t0 = time.time()
n = 100
for _ in range(n):
    engine.simulate_tournament()
t1 = time.time()

elapsed = t1 - t0
print(f"Simulated {n} tournaments in {elapsed:.4f} seconds.")
print(f"Estimated time for 3,000 simulations: {elapsed * 30:.2f} seconds.")
