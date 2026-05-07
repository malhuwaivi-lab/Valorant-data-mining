import pandas as pd

# Load the raw dataset that was created after merging all years and all selected files
df = pd.read_csv("merged_raw_all_years.csv")

print("Original shape:", df.shape)

#duplicate check
print("\nDuplicate rows:", df.duplicated().sum())

#Remove duplicate rows from the dataset if exist
df = df.drop_duplicates()

print("Shape after dropping duplicates:", df.shape)

#Show columns that contain missing values before preprocessing
print("\nMissing values before cleaning:")
missing_before = df.isnull().sum()
print(missing_before[missing_before > 0].sort_values(ascending=False))

#Remove rows where the target value is missing
df = df.dropna(subset=["win"])

#Convert the target column to integer values
df["win"] = df["win"].astype(int)

#Select columns for the final dataset
final_cols = [
    "year", "Tournament", "Stage", "Match Type", "Match Name", "Map",
    "Team", "opponent",

    #These score columns are kept in the cleaned dataset for reference (but not for used while training as we said)
    "team_score", "opponent_score", "score_diff",
    "attacker_score", "defender_score", "overtime_score",

    #Player performance features
    "avg_rating", "avg_acs",
    "total_kills", "total_deaths", "total_assists",
    "kd_diff", "avg_kast", "avg_adr", "avg_headshot_pct",
    "total_first_kills", "total_first_deaths", "fkd_diff",

    #Target feature
    "win"
]

#Add economy columns automatically, these were created from eco_stats after pivoting the economy Type column
eco_cols = [
    col for col in df.columns
    if col.startswith("Initiated_") or col.startswith("Won_")
]

#Insert economy columns before the target column
final_cols = final_cols[:-1] + eco_cols + final_cols[-1:]

#Keep only columns that actually exist in the dataset
final_cols = [col for col in final_cols if col in df.columns]

#Create the cleaned dataset using the selected columns
df_clean = df[final_cols].copy()

#Fill missing values
#First separate numeric and text columns
numeric_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns
text_cols = df_clean.select_dtypes(include=["object"]).columns

#Fill missing numeric values using the median of each column
for col in numeric_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

#Fill missing text values with "Unknown"
for col in text_cols:
    df_clean[col] = df_clean[col].fillna("Unknown")

#Print the final shape after preprocessing
print("\nFinal shape:", df_clean.shape)

#Check if any missing values remain
print("\nMissing values after cleaning:")
print(df_clean.isnull().sum().sum())

#Check the distribution of the target column
print("\nTarget distribution:")
print(df_clean["win"].value_counts())

#Print final column names
print("\nColumns:")
print(df_clean.columns.tolist())

# Save the cleaned dataset as a CSV file
df_clean.to_csv("cleaned_win_prediction_dataset.csv", index=False)

print("\nSaved: cleaned_win_prediction_dataset.csv")