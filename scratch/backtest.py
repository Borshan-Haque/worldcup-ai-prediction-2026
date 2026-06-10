"""
WorldCupAI — Backtesting and Validation Engine
Evaluates the predictive accuracy of the baseline vs upgraded forecasting engine
on historical FIFA World Cups (2010, 2014, 2018, 2022).
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from utils.engine import FootballEngine

# ---------------------------------------------------------------------------
# Historical Tournament Configurations
# ---------------------------------------------------------------------------
TOURNAMENTS = {
    2010: {
        "start_date": "2010-06-11",
        "end_date": "2010-07-11",
        "cutoff_date": "2010-06-10",
        "hosts": {"South Africa"},
        "reigning_champion": "Italy",
        "first_timers": {"Slovakia"}
    },
    2014: {
        "start_date": "2014-06-12",
        "end_date": "2014-07-13",
        "cutoff_date": "2014-06-11",
        "hosts": {"Brazil"},
        "reigning_champion": "Spain",
        "first_timers": {"Bosnia and Herzegovina"}
    },
    2018: {
        "start_date": "2018-06-14",
        "end_date": "2018-07-15",
        "cutoff_date": "2018-06-13",
        "hosts": {"Russia"},
        "reigning_champion": "Germany",
        "first_timers": {"Iceland", "Panama"}
    },
    2022: {
        "start_date": "2022-11-20",
        "end_date": "2022-12-18",
        "cutoff_date": "2022-11-19",
        "hosts": {"Qatar"},
        "reigning_champion": "France",
        "first_timers": {"Qatar"}
    }
}

def evaluate_predictions(engine: FootballEngine, year: int, config: dict):
    print(f"\n--- Backtesting World Cup {year} ---")
    cutoff = pd.Timestamp(config["cutoff_date"])
    start = pd.Timestamp(config["start_date"])
    end = pd.Timestamp(config["end_date"])
    
    # 1. Compute ratings up to the cutoff date (no future leak)
    engine.compute_ratings(cutoff_date=cutoff)
    
    # 2. Get matches for this World Cup
    df = engine._results_df
    wc_matches = df[
        (df["tournament"] == "FIFA World Cup") & 
        (df["date"] >= start) & 
        (df["date"] <= end)
    ].copy()
    
    print(f"Found {len(wc_matches)} matches to evaluate.")
    if len(wc_matches) == 0:
        return None
        
    metrics = {
        "baseline": {"log_loss": [], "brier": [], "correct": 0},
        "upgraded": {"log_loss": [], "brier": [], "correct": 0}
    }
    
    for _, row in wc_matches.iterrows():
        home, away = row["home_team"], row["away_team"]
        try:
            hs, as_ = int(row["home_score"]), int(row["away_score"])
        except (ValueError, TypeError):
            continue
            
        # Determine actual outcome: 0=Home Win, 1=Draw, 2=Away Win
        if hs > as_:
            actual = 0
        elif hs == as_:
            actual = 1
        else:
            actual = 2
            
        y_true = np.zeros(3)
        y_true[actual] = 1.0
        
        for mode in ["baseline", "upgraded"]:
            pred = engine.predict_match(
                home=home,
                away=away,
                venue=None, # Venue factors are handled differently in backtest
                mode=mode,
                tournament_hosts=config["hosts"],
                reigning_champion=config["reigning_champion"],
                first_timers=config["first_timers"]
            )
            
            p_home = pred["home_win_pct"] / 100.0
            p_draw = pred["draw_pct"] / 100.0
            p_away = pred["away_win_pct"] / 100.0
            
            # Clip probabilities to avoid log(0)
            eps = 1e-15
            p = np.clip([p_home, p_draw, p_away], eps, 1 - eps)
            p /= p.sum() # renormalize
            
            # Log Loss
            log_loss = -np.sum(y_true * np.log(p))
            metrics[mode]["log_loss"].append(log_loss)
            
            # Brier Score
            brier = np.sum((p - y_true) ** 2)
            metrics[mode]["brier"].append(brier)
            
            # Accuracy
            pred_outcome = np.argmax(p)
            if pred_outcome == actual:
                metrics[mode]["correct"] += 1

    summary = {}
    for mode in ["baseline", "upgraded"]:
        n_matches = len(metrics[mode]["log_loss"])
        avg_loss = np.mean(metrics[mode]["log_loss"]) if n_matches else 0.0
        avg_brier = np.mean(metrics[mode]["brier"]) if n_matches else 0.0
        accuracy = (metrics[mode]["correct"] / n_matches) * 100.0 if n_matches else 0.0
        summary[mode] = {
            "log_loss": avg_loss,
            "brier": avg_brier,
            "accuracy": accuracy
        }
        print(f"[{mode.upper()}] Log Loss: {avg_loss:.4f} | Brier: {avg_brier:.4f} | Accuracy: {accuracy:.2f}%")
        
    return summary

def main():
    engine = FootballEngine()
    engine.load_data()
    
    results = {}
    for year, config in TOURNAMENTS.items():
        results[year] = evaluate_predictions(engine, year, config)
        
    # Print overall summary
    print("\n" + "="*50)
    print("                 OVERALL BACKTEST SUMMARY")
    print("="*50)
    
    overall = {"baseline": {"log_loss": [], "brier": [], "acc": []},
               "upgraded": {"log_loss": [], "brier": [], "acc": []}}
               
    for year, res in results.items():
        if res is None:
            continue
        for mode in ["baseline", "upgraded"]:
            overall[mode]["log_loss"].append(res[mode]["log_loss"])
            overall[mode]["brier"].append(res[mode]["brier"])
            overall[mode]["acc"].append(res[mode]["accuracy"])
            
    print(f"{'Metric':<15} | {'Baseline':<12} | {'Upgraded':<12} | {'Improvement':<12}")
    print("-"*60)
    for metric_name, key in [("Log Loss", "log_loss"), ("Brier Score", "brier"), ("Accuracy", "acc")]:
        val_base = np.mean(overall["baseline"][key])
        val_up = np.mean(overall["upgraded"][key])
        if key == "acc":
            diff = val_up - val_base
            sign = "+" if diff > 0 else ""
            print(f"{metric_name:<15} | {val_base:.2f}%      | {val_up:.2f}%      | {sign}{diff:.2f}%")
        else:
            diff = val_base - val_up # for loss/brier, lower is better, so positive means improvement
            sign = "+" if diff > 0 else ""
            print(f"{metric_name:<15} | {val_base:.4f}       | {val_up:.4f}       | {sign}{diff:.4f}")
            
if __name__ == "__main__":
    main()
