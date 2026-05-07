import pandas as pd

#Load the combined CSV files that were created from all VCT years
maps_scores = pd.read_csv("combined_data/all_maps_scores.csv")
overview = pd.read_csv("combined_data/all_overview.csv")
eco = pd.read_csv("combined_data/all_eco_stats.csv")

#Print the shape of each dataset to check the number of rows and columns
print("maps_scores shape:", maps_scores.shape)
print("overview shape:", overview.shape)
print("eco shape:", eco.shape)

#Print column names to understand the structure of each file
print("\nMaps columns:")
print(maps_scores.columns.tolist())

print("\nOverview columns:")
print(overview.columns.tolist())

print("\nEco columns:")
print(eco.columns.tolist())

#Convert maps_scores to team-level rows
#Each map has two teams in the same row: Team A and Team B.
#we convert each map into two rows: one row for Team A and one row for Team B.

#Create rows for Team A
team_a = maps_scores[[
    "year", "Tournament", "Stage", "Match Type", "Match Name", "Map",
    "Team A", "Team A Score", "Team A Attacker Score",
    "Team A Defender Score", "Team A Overtime Score",
    "Team B", "Team B Score"
]].copy()

#Rename Team A columns into general team-level column names
team_a = team_a.rename(columns={
    "Team A": "Team",
    "Team A Score": "team_score",
    "Team A Attacker Score": "attacker_score",
    "Team A Defender Score": "defender_score",
    "Team A Overtime Score": "overtime_score",
    "Team B": "opponent",
    "Team B Score": "opponent_score"
})

#Create rows for Team B
team_b = maps_scores[[
    "year", "Tournament", "Stage", "Match Type", "Match Name", "Map",
    "Team B", "Team B Score", "Team B Attacker Score",
    "Team B Defender Score", "Team B Overtime Score",
    "Team A", "Team A Score"
]].copy()

#Rename Team B columns into the same general team-level column names
team_b = team_b.rename(columns={
    "Team B": "Team",
    "Team B Score": "team_score",
    "Team B Attacker Score": "attacker_score",
    "Team B Defender Score": "defender_score",
    "Team B Overtime Score": "overtime_score",
    "Team A": "opponent",
    "Team A Score": "opponent_score"
})

#Combine Team A rows and Team B rows into one team-level dataset
score_long = pd.concat([team_a, team_b], ignore_index=True)

#Create the target variable: win = 1 if the team score is higher than the opponent score, otherwise 0
score_long["win"] = (score_long["team_score"] > score_long["opponent_score"]).astype(int)

#Create score difference as an additional raw feature (will not be used during training)
score_long["score_diff"] = score_long["team_score"] - score_long["opponent_score"]

#Check the new team-level score dataset
print("\nscore_long shape:", score_long.shape)
print(score_long.head())

#Clean overview numeric columns
#Some percentage columns are stored as text with a % sign. So we remove the % sign and convert them to numeric values.
for col in ["Kill, Assist, Trade, Survive %", "Headshot %"]:
    if col in overview.columns:
        overview[col] = overview[col].astype(str).str.replace("%", "", regex=False)
        overview[col] = pd.to_numeric(overview[col], errors="coerce")

#List of overview columns that should be numeric
numeric_cols = [
    "Rating", "Average Combat Score", "Kills", "Deaths", "Assists",
    "Kills - Deaths (KD)", "Kill, Assist, Trade, Survive %",
    "Average Damage Per Round", "Headshot %",
    "First Kills", "First Deaths", "Kills - Deaths (FKD)"
]

#Convert all selected columns to numeric values
for col in numeric_cols:
    if col in overview.columns:
        overview[col] = pd.to_numeric(overview[col], errors="coerce")

#Aggregate player overview to team-level
#overview.csv is player-level data. So we group players by match, map, and team to create team-level performance features.
group_cols = ["year", "Tournament", "Stage", "Match Type", "Match Name", "Map", "Team"]

team_overview = overview.groupby(group_cols).agg(
    avg_rating=("Rating", "mean"),
    avg_acs=("Average Combat Score", "mean"),
    total_kills=("Kills", "sum"),
    total_deaths=("Deaths", "sum"),
    total_assists=("Assists", "sum"),
    kd_diff=("Kills - Deaths (KD)", "sum"),
    avg_kast=("Kill, Assist, Trade, Survive %", "mean"),
    avg_adr=("Average Damage Per Round", "mean"),
    avg_headshot_pct=("Headshot %", "mean"),
    total_first_kills=("First Kills", "sum"),
    total_first_deaths=("First Deaths", "sum"),
    fkd_diff=("Kills - Deaths (FKD)", "sum")
).reset_index()

#Check the aggregated team-level overview dataset
print("\nteam_overview shape:", team_overview.shape)

#Pivot eco_stats
#eco_stats contains economy types as rows. So we pivot the Type column to convert economy categories into separate columns.
eco_pivot = eco.pivot_table(
    index=["year", "Tournament", "Stage", "Match Type", "Match Name", "Map", "Team"],
    columns="Type",
    values=["Initiated", "Won"],
    aggfunc="sum"
)

#Flatten the multi-level column names created by pivot_table
eco_pivot.columns = [
    f"{value}_{eco_type}".replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    for value, eco_type in eco_pivot.columns
]

#Reset the index to make the grouping columns normal columns again
eco_pivot = eco_pivot.reset_index()

#Check the economy feature dataset
print("eco_pivot shape:", eco_pivot.shape)

# Merge score data with aggregated team performance data
merged = score_long.merge(
    team_overview,
    on=["year", "Tournament", "Stage", "Match Type", "Match Name", "Map", "Team"],
    how="left"
)

#Merge the result with economy features
merged = merged.merge(
    eco_pivot,
    on=["year", "Tournament", "Stage", "Match Type", "Match Name", "Map", "Team"],
    how="left"
)

#Save the final raw merged dataset before preprocessing
merged.to_csv("merged_raw_all_years.csv", index=False)

#Print final information about the merged dataset
print("\nFinal merged shape:", merged.shape)

print("\nMissing values:")
print(merged.isnull().sum())

print("\nSaved: merged_raw_all_years.csv")