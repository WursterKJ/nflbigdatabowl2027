import numpy as np
import pandas as pd

path = "data/"

players = pd.read_csv(f"{path}players.csv")
combine_results = pd.read_csv(f"{path}combine_results.csv")
player_career_successes = pd.read_csv(f"{path}player_career_successes.csv")
player_play = pd.read_csv(f"{path}player_play.csv")
games = pd.read_csv(f"{path}games.csv")
"""
combine_tracking = pd.read_csv(f"{path}combine_tracking.csv")
game_tracking_2023 = pd.read_csv(f"{path}game_tracking_2023.csv")
game_tracking_2024 = pd.read_csv(f"{path}game_tracking_2024.csv")
game_tracking_2025 = pd.read_csv(f"{path}game_tracking_2025.csv")
"""


# exploring positional breakdowns
players_df = players.merge(player_career_successes, how="left", on='nfl_id')
players_df = players_df.merge(combine_results, how="left", on="nfl_id")

positions = players_df["nfl_position"].unique()
print(positions)

"""
draft_positions = []
for pos in positions:
    pos_df=players_df[players_df["nfl_position"] == pos]
    draft_positions.append({
        "position":pos,
        "players":len(pos_df),
        "drafted":pos_df["draft_overall_pick"].notna().sum(),
        "undrafted":pos_df["draft_overall_pick"].isna().sum(),
        "avg_pick":pos_df["draft_overall_pick"].mean(),
    })
summary = pd.DataFrame(draft_positions).sort_values("avg_pick").reset_index(drop=True)
print(summary)
"""

# epa per target WRs
plays = player_play.merge(players, how='left', on='nfl_id').merge(games, how='left', on='game_id')
reg = plays[plays["season_type"]=="REG"]
wr_plays = reg[reg["nfl_position"]=="WR"]
wr_targeted = wr_plays[wr_plays["target"].astype(bool)]
wr_epa = (wr_targeted.groupby(["nfl_id", "display_name"], as_index=False).agg(targets=("target", "size"), epa_per_target=("expected_points_added", "mean")).drop(columns="nfl_id").sort_values("epa_per_target", ascending=False))
print(wr_epa.to_string())


