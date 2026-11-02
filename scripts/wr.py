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
games_t_23 = pd.read_csv(f"{path}game_tracking_2023.csv")
games_t_24 = pd.read_csv(f"{path}game_tracking_2024.csv")
games_t_25 = pd.read_csv(f"{path}game_tracking_2025.csv")

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
#     print(profile_draft)
# wr.to_csv(f"{path}wr_zscores.csv")

# modeling for hook, find marginal changes in score to draft pick
model = smf.ols("pick_log ~ z_ath + z_prod", wr).fit()
# print(model.summary())
# print(model.t_test("z_ath - z_prod = 0"))



# create outcome/performance variables for WRs
# combine performance outcomes with draft position for player profiles
# aggregate per route/target metrics two different ways: (example for epa per route run)
     # sum all profiled players epa's then divide by total routes ran for all players in profile
     # create metrics on individual level, then average across profiles (epa per route run for each player, then average)
plays = pd.read_csv(f"{path}player_play.csv")
games = pd.read_csv(f"{path}games.csv")
wr_plays = plays[plays["nfl_id"].isin(wr_ids) & (plays["play_nullified_by_penalty"] == "N")]
wr_plays = games[games["season_type"] == "REG"].merge(wr_plays, on="game_id", how="inner")
wr_plays["yac_over_expected"] = wr_plays["yards_after_catch"] - wr_plays["expected_yards_after_catch"]
wr_plays[["rec_yards", "yac_over_expected", "expected_points_added"]] = wr_plays[["rec_yards", "yac_over_expected", "expected_points_added"]].fillna(0)
wr_plays = wr_plays.merge(wr[["nfl_id", "profile"]], on="nfl_id", how="left")
wr_targeted = wr_plays[wr_plays["target"] == True]

# first aggregation, profile level (pool all plays by profile first, then make metrics)
wr_profile_draft = wr.groupby("profile").agg(
    Avg_Pick=("pick_adj", "mean"),
    Med_Pick=("pick_adj", "median"),
    Avg_Pick_Drafted=("draft_overall_pick", "mean"),
    UDFA_Rate=("undrafted", "mean"),
    Snaps=("career_offensive_snaps", "sum"),
    Games_Started=("career_games_started", "sum"),
    AllPro_First=("ap_all_pro_1st_team", "sum"),
    AllPro_Second=("ap_all_pro_2nd_team", "sum"),
    Pro_Bowl=("pro_bowl_original_ballot", "sum"))
wr_profile_routes = wr_plays.groupby("profile").agg(
    routes=("play_id", "count"),
    epa_route=("expected_points_added", "mean"),
    yprr=("rec_yards", "mean"),
    yac_over_exp_route=("yac_over_expected", "mean"))
wr_profile_targets = wr_targeted.groupby("profile").agg(
    targets=("play_id", "count"),
    epa_target=("expected_points_added", "mean"),
    ypt=("rec_yards", "mean"),
    yac_over_exp_target=("yac_over_expected", "mean"))
wr_pooled = wr_profile_draft.join(wr_profile_routes).join(wr_profile_targets).sort_values("Avg_Pick")
wr_pooled["aggregation"] = "profile"

# second aggregation, player level (average players metrics by profile)
wr_player_routes = wr_plays.groupby("nfl_id").agg(
    routes=("play_id", "count"),
    epa_route=("expected_points_added", "mean"),
    yprr=("rec_yards", "mean"),
    yac_over_exp_route=("yac_over_expected", "mean"))
wr_player_targets = wr_targeted.groupby("nfl_id").agg(
    targets=("play_id", "count"),
    epa_target=("expected_points_added", "mean"),
    ypt=("rec_yards", "mean"),
    yac_over_exp_target=("yac_over_expected", "mean"))
wr_player = (wr_player_routes.join(wr_player_targets).join(wr.set_index("nfl_id")["profile"]))
wr_player_avg = wr_player.groupby("profile").agg(
    routes=("routes", "sum"),
    epa_route=("epa_route", "mean"),
    yprr=("yprr", "mean"),
    yac_over_exp_route=("yac_over_exp_route", "mean"),
    targets=("targets", "sum"),
    epa_target=("epa_target", "mean"),
    ypt=("ypt", "mean"),
    yac_over_exp_target=("yac_over_exp_target", "mean"))
wr_player_avg["aggregation"] = "player"
pick_cols = ["Avg_Pick", "Med_Pick", "Avg_Pick_Drafted", "UDFA_Rate", "Snaps", "Games_Started", "AllPro_First", "AllPro_Second", "Pro_Bowl"]
wr_player_avg = wr_player_avg.join(wr_profile_draft[pick_cols]).loc[wr_pooled.index]
wr_player_avg["aggregation"] = "player avg"

wr_profile_perform = pd.concat([wr_pooled, wr_player_avg]).reset_index()
first_cols = ["aggregation", "profile"]
wr_profile_perform = wr_profile_perform[first_cols + [c for c in wr_pooled.columns if c not in first_cols]].round(3)
print(wr_profile_perform.to_string(index=False))
wr_profile_perform.to_csv(f"{path}wr_profile_perform.csv", index=False)