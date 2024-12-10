import pandas as pd
from src.analysis import write_season

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, data_2, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """

    mvp_table, _ = data
    player_stats = data_2

    for player in list(mvp_table['Player']):
        mvp_table = add_player_stats_to_dataframe(mvp_table, player, player_stats[player]['stats'][0])
        mvp_table = add_player_stats_to_dataframe(mvp_table, player, player_stats[player]['advanced'][0])

    return mvp_table


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'


def add_player_stats_to_dataframe(df, player_name, stats_dict, exclude_keys=None):
    """
    Adds stats from a dictionary as new columns to a dataframe row corresponding to a player's name.
    
    Parameters:
        df (pd.DataFrame): The dataframe to modify. Must contain a 'Player' column.
        player_name (str): The player's name to match in the dataframe.
        stats_dict (dict): Dictionary of stats to add as columns.
        exclude_keys (list): List of keys to exclude from the stats_dict. Default keys are excluded.
    
    Returns:
        pd.DataFrame: The updated dataframe.
    """
    if exclude_keys is None:
        exclude_keys = ["Season", "Age", "Team", "Lg", "Pos", "G", "GS", "MP", "Awards"]
    
    # Filter out excluded keys
    filtered_stats = {key: value for key, value in stats_dict.items() if key not in exclude_keys}
    
    # Ensure the 'Player' column exists
    if "Player" not in df.columns:
        raise ValueError("The dataframe must contain a 'Player' column.")
    
    # Find the row index for the player
    matching_rows = df[df["Player"] == player_name]
    if matching_rows.empty:
        raise ValueError(f"Player '{player_name}' not found in the dataframe.")
    
    row_index = matching_rows.index[0]  # Get the first matching row's index
    
    # Add new columns if they don't exist
    for key in filtered_stats:
        if key not in df.columns:
            df[key] = None  # Add new column initialized with None
    
    # Update the values in the row for the filtered stats
    for key, value in filtered_stats.items():
        df.at[row_index, key] = value
    
    return df
