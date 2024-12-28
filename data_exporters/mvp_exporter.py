from mage_ai.io.file import FileIO
from pandas import DataFrame
from src.analysis import write_season

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data_to_file(df: DataFrame, **kwargs) -> None:
    """
    Template for exporting data to filesystem.

    Docs: https://docs.mage.ai/design/data-loading#fileio
    """

    season = kwargs.get('season', None)
    formatted_season = write_season(season)

    filepath = f'/Users/cb/src/nba_mvp_ml/data/{formatted_season}.csv'
    FileIO().export(df, filepath)
