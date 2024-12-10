import pandas as pd
import requests
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
    
# before running on a new system: export PYTHONPATH=/Users/<USER>/<REPO_DIR>/nba_mvp_ml:$PYTHONPATH
from src.analysis import fix_encoding, write_season
import time

@data_loader
def load_data_from_api(*args, **kwargs) -> pd.DataFrame:
    """
    Fetch MVP voting data for a specific season from the Basketball Reference website.

    Args:
        season (int): The year of the NBA season to fetch data for (e.g., 2024 for the 2023-24 season).

    Returns:
        pd.DataFrame: A DataFrame containing the MVP voting data for the specified season.
    """
    season = kwargs.get('season', None)
    formatted_season = write_season(season)
    print(f"Fetching MVP data for season: {formatted_season}")
    url = f"https://www.basketball-reference.com/awards/awards_{season+1}.html"

    try:
        # Make GET request to fetch the HTML page
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse HTML to extract tables
        tables = pd.read_html(response.text)

        # Assume the MVP voting table is the first table on the page
        mvp_table = tables[0]
        mvp_table.columns = mvp_table.columns.get_level_values(1)
    
        # Ensure column names are strings
        mvp_table.columns = [str(col).replace(" ", "_").replace("Tm", "team") for col in mvp_table.columns]
        # mvp_table.columns = [col.lower() for col in mvp_table.columns]
        mvp_table['Player'] = mvp_table['Player'].apply(fix_encoding)

        # Reset index and ensure it is compatible with JSON
        mvp_table.reset_index(drop=True, inplace=True)

        # Append the season for later filtering
        mvp_table["season"] = season

        print(f"Successfully fetched data for season: {formatted_season}")
        time.sleep(2)
        
        return mvp_table, response.text

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for season {season}: {e}")
        raise
    except ValueError as e:
        print(f"Error parsing HTML tables for season {season}: {e}")
        raise

@test
def test_output(output: pd.DataFrame, *args) -> None:
    """
    Test to validate the output of the MVP voting data fetch block.

    Args:
        output (pd.DataFrame): The output DataFrame to validate.
    """
    assert output is not None, 'The output is undefined'
    assert isinstance(output, pd.DataFrame), 'Output should be a pandas DataFrame'
    assert not output.empty, 'Output DataFrame is empty'

    # Check if all columns are strings
    assert all(isinstance(col, str) for col in output.columns), 'All column names must be strings'

    # Check if the index is reset and numeric
    assert output.index.is_integer(), 'Index must be numeric'

    print("All tests passed successfully!")