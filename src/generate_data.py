import pandas as pd
import random
from pathlib import Path

num_games = 600

data = []

for _ in range(num_games):
    home = random.randint(0, 1)

    # Generate realistic football stats
    points_scored = random.randint(7, 40)
    points_allowed = random.randint(7, 40)
    total_yards = random.randint(250, 450)
    turnovers = random.randint(0, 4)

    win = 1 if points_scored > points_allowed else 0

    data.append([
        points_scored,
        points_allowed,
        total_yards,
        turnovers,
        home,
        win
    ])

df = pd.DataFrame(
    data,
    columns=[
        "points_scored",
        "points_allowed",
        "total_yards",
        "turnovers",
        "home",
        "win"
    ]
)

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"
data_dir.mkdir(exist_ok=True)

df.to_csv(data_dir / "jets_games.csv", index=False)

print("✅ Dataset generated with", num_games, "games.")