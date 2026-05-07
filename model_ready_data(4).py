import pandas as pd

df = pd.read_csv("cleaned_win_prediction_dataset.csv")

leakage_cols = [
    "team_score",
    "opponent_score",
    "score_diff"
]

df_model = df.drop(columns=[col for col in leakage_cols if col in df.columns])

df_model.to_csv("model_ready_win_prediction_dataset.csv", index=False)

print("Saved: model_ready_win_prediction_dataset.csv")
print("Shape:", df_model.shape)
print("Columns:")
print(df_model.columns.tolist())