import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

path = "data/"

# WR descriptive data
combine = pd.read_csv(f"{path}combine_results.csv")
player_accol = pd.read_csv(f"{path}player_career_successes.csv")
players = pd.read_csv(f"{path}players.csv")
wr = players[players["nfl_position"] == "WR"]
wr = wr.merge(player_accol, how="left", on="nfl_id").merge(combine.drop(columns="draft_year"), how="left", on="nfl_id")
wr_ids = wr["nfl_id"]

# WR combine tracking data
combine_t = pd.read_csv(f"{path}combine_tracking.csv")
wr_combine_t = combine_t[combine_t["nfl_id"].isin(wr_ids)]

# WR game tracking data
plays = pd.read_csv(f"{path}player_play.csv")
games = pd.read_csv(f"{path}games.csv")
games_t_23 = pd.read_csv(f"{path}game_tracking_2023.csv")
games_t_24 = pd.read_csv(f"{path}game_tracking_2024.csv")
games_t_25 = pd.read_csv(f"{path}game_tracking_2025.csv")

wr_plays = plays[plays["nfl_id"].isin(wr_ids)]
wr_plays = games[games["season_type"] == "REG"].merge(wr_plays, on="game_id", how="inner")
game_ids = wr_plays["game_id"].unique()
wr_plays_t = pd.concat([track[track["game_id"].isin(game_ids) & track["nfl_id"].isin(wr_ids)]
     for track in (games_t_23, games_t_24, games_t_25)], ignore_index=True)

# wr = WR descriptive data such as measurables, combine measurements, accolades, ngs scores
# wr_combine_t = combine tracking for each WR
# wr_plays = regular season only play level data with game and play result information
# wr_plays_t = all tracking frames from plays from last three regular seasons stacked only for WRs
