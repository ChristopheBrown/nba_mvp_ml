import pandas as pd
import requests
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

# before running on a new system: export PYTHONPATH=/Users/<USER>/<REPO_DIR>/nba_mvp_ml:$PYTHONPATH
from src.analysis import *

@data_loader
def load_team_data(data, *args, **kwargs) -> pd.DataFrame:
    """
    Fetch team statistics for a specific NBA season from Basketball Reference.

    Args:
        year (int): The NBA season year to fetch data for (e.g., 2023 for the 2022-23 season).

    Returns:
        pd.DataFrame: A DataFrame containing "Team and Opponent Stats" for the given season.
    """
    season = kwargs.get('season', None)
    formatted_season = write_season(season)
    
    mvp_df = data

    try:
        mvp_df = append_team_stats(mvp_df, season)

        print(f"Successfully fetched team data for season: {formatted_season}")
        return mvp_df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for season {season}: {e}")
        raise
    except ValueError as e:
        print(f"Error parsing HTML tables for season {season}: {e}")
        raise

@test
def test_output(output: pd.DataFrame, *args) -> None:
    """
    Test to validate the output of the team data fetch block.

    Args:
        output (pd.DataFrame): The output DataFrame to validate.
    """
    assert output is not None, 'The output is undefined'
    assert isinstance(output, pd.DataFrame), 'Output should be a pandas DataFrame'
    assert not output.empty, 'Output DataFrame is empty'

    # Check if column names are strings
    assert all(isinstance(col, str) for col in output.columns), 'All column names must be strings'

    # Check if the index is numeric
    assert output.index.is_integer(), 'Index must be numeric'

    print("All tests passed successfully!")


def append_team_stats(mvp_df, season):
    # Define the columns to extract and prefix
    columns_to_add_misc = ["W", "L", "PW", "PL", "MOV", "SOS", "SRS", "ORtg", "DRtg", "Pace"]
    prefixed_columns_misc = {col: f"team_{col}" for col in columns_to_add_misc}

    columns_to_add_opp = ["G", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]
    prefixed_columns_opp = {col: f"opp_{col}" for col in columns_to_add_opp}

    # Iterate through each row and add the relevant team stats
    for index, row in mvp_df.iterrows():
        team = row["team"]
        print(f"Fetching team data ({team}) for season: {write_season(season)}")
        # Load team stats
        team_opps_df, misc_df = load_team_misc_and_opponent(season, team)
        # Extract the required columns and prefix them
        misc_df_subset = misc_df[columns_to_add_misc].rename(columns=prefixed_columns_misc)
        opp_df_subset = misc_df[columns_to_add_opp].rename(columns=prefixed_columns_opp)

        # Append the team stats to the player's row
        for col, value in misc_df_subset.iloc[0].items():
            mvp_df.at[index, col] = value

        # Append the team stats to the player's row
        for col, value in opp_df_subset.iloc[0].items():
            mvp_df.at[index, col] = value    

    return mvp_df