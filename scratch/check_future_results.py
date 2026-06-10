import pandas as pd

df = pd.read_csv("data/results.csv")
df["date"] = pd.to_datetime(df["date"])
future = df[df["date"] >= "2026-05-01"]
print("Matches after 2026-05-01:")
print(future.head(20))
print("Total future matches in results.csv:", len(future))
