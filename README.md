# WorldCupAI ⚽

AI-powered FIFA World Cup 2026 prediction & simulation platform.

## Setup

```bash
pip install -r requirements.txt
python train.py
streamlit run dashboard/app.py
```

## Features
- **Match Predictor** — Poisson + Elo xG model with score probabilities
- **Group Stage** — Simulate all 12 groups and predict standings
- **Knockout Bracket** — Full tournament simulation with ET & penalties
- **Team Analytics** — Radar charts, H2H simulation, tournament path
- **Monte Carlo** — Run up to 10,000 full World Cup simulations

## Data Sources
- `results.csv` — 49,000+ international match results
- `goalscorers.csv` — Individual goalscorer data
- `shootouts.csv` — Penalty shootout results
- `former_names.csv` — Country name history for name resolution
- `fifa-match-schedule.pdf` — FIFA World Cup 2026 schedule
