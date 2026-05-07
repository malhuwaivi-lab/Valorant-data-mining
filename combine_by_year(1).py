import pandas as pd
from pathlib import Path

#Main folder that contains all folders
DATA_DIR = Path("data")

#List of VCT yearly folders
YEARS = ["vct_2021", "vct_2022", "vct_2023", "vct_2024", "vct_2025", "vct_2026"]

#Files that need to be combined across all years
FILES_TO_COMBINE = {
    "maps_scores": "matches/maps_scores.csv",
    "eco_stats": "matches/eco_stats.csv",
    "overview": "matches/overview.csv"
}

#Output folder where the combined CSV files will be saved
OUTPUT_DIR = Path("combined_data")
OUTPUT_DIR.mkdir(exist_ok=True)

#Loop through each file type and combine it across all years
for output_name, relative_path in FILES_TO_COMBINE.items():
    dfs = []

    print("\n" + "=" * 60)
    print(f"Combining: {output_name}")
    print("=" * 60)

    #Loop through each year folder
    for year_folder in YEARS:
        file_path = DATA_DIR / year_folder / relative_path

        #Check if the file exists
        if file_path.exists():
            df = pd.read_csv(file_path)

            #Add a year column, for example: vct_2025 becomes 2025
            df["year"] = int(year_folder.replace("vct_", ""))

            #Store the dataframe in the list
            dfs.append(df)

            print(f"Found {file_path} | Shape: {df.shape}")
        else:
            print(f"Missing: {file_path}")

    #Combine all available yearly files into one dataframe
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)

        #Save the combined data as CSV
        output_path = OUTPUT_DIR / f"all_{output_name}.csv"
        combined_df.to_csv(output_path, index=False)

        print(f"\nSaved: {output_path}")
        print("Final shape:", combined_df.shape)
        print("Columns:")
        print(combined_df.columns.tolist())
    else:
        print(f"No files found for {output_name}")