import pandas as pd

res = pd.read_csv("data/results.csv")
goal = pd.read_csv("data/goalscorers.csv")

print("results.csv:")
print("  Total rows:", len(res))
print("  Max date:", res["date"].max())
print("  Min date:", res["date"].min())

print("\ngoalscorers.csv:")
print("  Total rows:", len(goal))
print("  Max date:", goal["date"].max())
print("  Min date:", goal["date"].min())
