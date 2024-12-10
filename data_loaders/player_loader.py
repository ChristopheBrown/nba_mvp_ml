import io
from src.analysis import get_player_stats_from_mvp_html, write_season

import requests
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data_from_api(data, *args, **kwargs):
    """
    Template for loading data from API
    """
    season = kwargs.get('season', None)
    formatted_season = write_season(season)
    print(f"Fetching traditional and advanced data for season: {formatted_season}")

    mvp_table, html_text = data

    player_stats = {}

    for player_name in list(mvp_table['Player']):
        player_stats[player_name] = {}
        
        _, per_game_stats_df, adv_stats_df = get_player_stats_from_mvp_html(html_content=html_text, player_name=player_name)

        per_game_stats_df_season = per_game_stats_df.loc[per_game_stats_df['Season'] == formatted_season]
        advanced_stats_df_season = adv_stats_df.loc[adv_stats_df['Season'] == formatted_season]

        player_stats[player_name]['stats'] = per_game_stats_df_season
        player_stats[player_name]['advanced'] = advanced_stats_df_season

    return player_stats


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
