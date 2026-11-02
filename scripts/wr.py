import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import statsmodels.formula.api as smf

path = "data/"

# WR descriptive data
combine = pd.read_csv(f"{path}combine_results.csv")
player_accol = pd.read_csv(f"{path}player_career_successes.csv")
players = pd.read_csv(f"{path}players.csv")
wr = players[players["nfl_position"] == "WR"]
wr = wr.merge(player_accol, how="left", on="nfl_id").merge(combine.drop(columns="draft_year"), how="left", on="nfl_id")
wr_ids = wr["nfl_id"]

'''
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
'''
     
# wr = WR descriptive data such as measurables, combine measurements, accolades, ngs scores
# wr_combine_t = combine tracking for each WR
# wr_plays = regular season only play level data with game and play result information
# wr_plays_t = all tracking frames from plays from last three regular seasons stacked only for WRs



# NGS SCORES (PRODUCTION VS ATHELTIC) AND DRAFT PICK POSITION
wr = wr.rename(columns={"ngs_athleticism_score":"ngs_ath", "ngs_college_production_score":"ngs_prod"})
wr["undrafted"] = wr["draft_overall_pick"].isna().astype(int)
wr["pick_adj"] = wr["draft_overall_pick"].fillna(players["draft_overall_pick"].max()+1)
wr["pick_log"] = np.log(wr["pick_adj"])
# print(wr[["ngs_ath", "ngs_prod"]].describe())
ath_mean = round(wr["ngs_ath"].mean(), 3)
ath_med = round(wr["ngs_ath"].median(), 3)
ath_sd = round(wr["ngs_ath"].std(), 3)
prod_mean = round(wr["ngs_prod"].mean(), 3)
prod_med = round(wr["ngs_prod"].median(), 3)
prod_sd = round(wr["ngs_prod"].std(), 3)
# print(ath_mean, ath_med, ath_sd, prod_mean, prod_med, prod_sd)
# 73.651 | 75.0 | 11.155 | 68.113 | 68.0 | 8.267
wr["z_ath"] = round(((wr["ngs_ath"] - ath_mean) / ath_sd), 3)
wr["z_prod"] = round(((wr["ngs_prod"] - prod_mean) / prod_sd), 3)

high_ath = wr["z_ath"] > 0
high_prod = wr["z_prod"] > 0
wr["profile"] = np.select([high_ath & high_prod, high_ath & ~high_prod, ~high_ath & high_prod], ["Both", "Athlete", "Producer"], default="Neither")
# print(wr["profile"].value_counts())
drafted = wr[wr["undrafted"]==False]
for pool in [wr, drafted]:
     profile_draft = (pool.groupby("profile").agg(Players=("nfl_id", "count"),
          Avg_Pick=("pick_adj", "mean"),
          Med_Pick=("pick_adj", "median"),
          Avg_Pick_Drafted=("draft_overall_pick", "mean"),
          UDFA_Rate=("undrafted", "mean"))
         .sort_values("Avg_Pick")
         .round(3))
     print(profile_draft)
# wr.to_csv(f"{path}wr_zscores.csv")

# modeling for hook, find marginal changes in score to draft pick
model = smf.ols("pick_log ~ z_ath + z_prod", wr).fit()
print(model.summary())
print(model.t_test("z_ath - z_prod = 0"))





