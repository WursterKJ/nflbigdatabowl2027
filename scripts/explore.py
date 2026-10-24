import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

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

'''
# exploring positional breakdowns
players_df = players.merge(player_career_successes, how="left", on='nfl_id')
players_df = players_df.merge(combine_results, how="left", on="nfl_id")

positions = players_df["nfl_position"].unique()
print(positions)


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


# epa per target WRs
plays = player_play.merge(players, how='left', on='nfl_id').merge(games, how='left', on='game_id')
reg = plays[plays["season_type"]=="REG"]
wr_plays = reg[reg["nfl_position"]=="WR"]
wr_targeted = wr_plays[wr_plays["target"].astype(bool)]
wr_epa = (wr_targeted.groupby(["nfl_id", "display_name"], as_index=False).agg(targets=("target", "size"), epa_per_target=("expected_points_added", "mean")).drop(columns="nfl_id").sort_values("epa_per_target", ascending=False))
print(wr_epa.to_string())
'''
# explore production vs athleticism scores

player_combine = players.merge(combine_results, how='left', on='nfl_id')
player_combine_filter = player_combine[["nfl_id", "display_name", "combine_position", "draft_overall_pick", 
                                        "ngs_athleticism_score", "ngs_college_production_score", "ngs_final_score"]].rename(columns=
    {"ngs_athleticism_score":"ngs_ath", "ngs_college_production_score":"ngs_prod", "ngs_final_score":"ngs_final"})

# print(player_combine_filter)
combine_positions = player_combine_filter["combine_position"].unique()
# print(combine_positions)
player_combine_filter["ath_margin"]=player_combine_filter["ngs_ath"]-player_combine_filter["ngs_prod"]
descrp_cols = ["draft_overall_pick", "ngs_ath", "ngs_prod", "ngs_final", "ath_margin"]
pos_descrp = player_combine_filter.groupby("combine_position")[descrp_cols].describe().round(2)
# pos_descrp.to_csv(f"{path}pos_descrp.csv")

# create scatter plot
plot_df = player_combine_filter.dropna(subset=["ngs_ath", "ngs_prod"])
r2 = linregress(plot_df["ngs_ath"], plot_df["ngs_prod"]).rvalue ** 2

sns.scatterplot(data=plot_df, x="ngs_ath", y="ngs_prod", hue="combine_position")
sns.regplot(data=plot_df, x="ngs_ath", y="ngs_prod", scatter=False, color="black")
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.title(f"R2 = {r2:.3f}")
plt.show()

