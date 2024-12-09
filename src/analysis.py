import pandas as pd

import plotly.express as px
import plotly.io as pio
# pio.renderers.default = 'notebook'
pio.renderers.default = 'iframe'

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

from bs4 import BeautifulSoup, Comment
import requests
import time


# player_stats_path = f'/Users/cb/src/nba_mvp_ml/data/processed/by_season/players/players_{year}.csv'
# team_stats_path = f'/Users/cb/src/nba_mvp_ml/data/processed/by_season/team (basketball-reference)/team_stats_{year}_updated.csv'
# mvp_votes_path = f'/Users/cb/src/nba_mvp_ml/data/processed/by_season/mvp/sentiment/mvp_{year}-{str(year+1)[2:]}.csv'

player_stats_path = f'/Users/cb/src/nba_mvp_ml/data/processed/by_season/players'
team_stats_path = f'/Users/cb/src/nba_mvp_ml/data/processed/by_season/team (basketball-reference)'
mvp_votes_path = f'/Users/cb/src/nba_mvp_ml/data/processed/by_season/mvp/sentiment'

def write_season(year):
    return f'{year}-{str(year+1)[2:]}'

def fix_encoding(text):
    try:
        return text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text  # Return original text if fixing fails

def load_year(year, player_path=player_stats_path, team_path=team_stats_path, mvp_path=mvp_votes_path, debug=False):

    player_path=os.path.join(player_path,f'players_{year}.csv')
    team_path=os.path.join(team_path,f'team_stats_{year}_updated.csv')
    mvp_path=os.path.join(mvp_path,f'mvp_{write_season(year)}.csv')                       

    def fix_encoding(text):
        try:
            return text.encode('latin1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text  # Return original text if fixing fails
    
    player_df = pd.read_csv(player_path)
    team_df = pd.read_csv(team_path).dropna(axis=1, how='all')
    mvp_df = pd.read_csv(mvp_path)
    
    player_df['PLAYER_FULLNAME'] = player_df['PLAYER_FULLNAME'].apply(fix_encoding)
    mvp_df['Player'] = mvp_df['Player'].apply(fix_encoding)
    
    if debug:
        print(f'MVP List: {list(mvp_df['Player'])}\n')
        print(f'Player dataframe columns:\n{list(player_df.columns)}\n')
        print(f'Team dataframe columns:\n{list(team_df.columns)}\n')
        print(f'MVP dataframe columns:\n{list(mvp_df.columns)}\n')

    return player_df, team_df, mvp_df

def merge_dfs(player_df, team_df, mvp_df, include_non_mvp=False, debug=False):
    # Merge player and team data
    merged_df = pd.merge(player_df, team_df, on=['TEAM_ID', 'SEASON_ID'], suffixes=('_player', '_team'))

    # Add a column to differentiate MVP candidates
    merged_df['MVP_Candidate'] = merged_df['PLAYER_FULLNAME'].apply(
        lambda x: 'MVP Candidate' if x in mvp_df['Player'].values else 'Other'
    )

    # Merge MVP voting data into the player/team dataset
    merged_with_mvp = pd.merge(
        merged_df,
        mvp_df.drop(columns=['Age','Tm']),
        how='left',
        left_on='PLAYER_FULLNAME',
        right_on='Player'
    )

    # Replace metrics with MVP data where available, with fallback logic
    merged_with_mvp['WS'] = merged_with_mvp['WS_y'].fillna(merged_with_mvp['WS_x'])
    merged_with_mvp['PTS'] = merged_with_mvp['PTS'].fillna(merged_with_mvp['PTS_player'])
    merged_with_mvp['TRB'] = merged_with_mvp['TRB_y'].fillna(merged_with_mvp['REB'])
    merged_with_mvp['AST'] = merged_with_mvp.get('AST_y', merged_with_mvp['AST_player'])  # Fallback if 'AST_y' is missing

    # Optionally filter out non-MVP candidates
    if not include_non_mvp:
        merged_with_mvp = merged_with_mvp[merged_with_mvp['MVP_Candidate'] != 'Other']
    
    if debug:
        display(sorted(list(merged_with_mvp.columns)))
    
    # Drop unnecessary columns and avoid confusion between suffixes
    merged_with_mvp = merged_with_mvp.rename(columns={'Pts Won': 'Pts_Won'})

    return merged_with_mvp

# --------------------------------------------------------------------------------------------------------
# PLAYER STATS -------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------

def get_player_stats_from_mvp_html(html_content, player_name, base_url="https://www.basketball-reference.com/"):
    """
    Extracts the first table associated with the player's page.
    
    Parameters:
        html_content (str): The HTML content of the page.
        player_name (str): The name of the player to find.
        base_url (str): The base URL to construct the full player link if needed.

    Returns:
        pd.DataFrame: A DataFrame containing the first table from the player's page.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Step 1: Find the player's link in the HTML
    player_link = None
    for link in soup.find_all('a', href=True):
        link_text = fix_encoding(link.text.strip())
        if link_text == player_name:
            player_link = link['href']
            break
    
    if not player_link:
        raise ValueError(f"No hyperlink found for player: {player_name}")
    
    # If base_url is provided, construct the full URL
    if base_url and not player_link.startswith('http'):
        player_link = base_url.rstrip('/') + '/' + player_link.lstrip('/')
    
    # Step 2: Fetch the HTML content of the player's page
    player_response = requests.get(player_link)
    time.sleep(2)
    player_html_content = player_response.text
    
    # Step 3: Parse the player's page and extract the first table
    player_soup = BeautifulSoup(player_html_content, 'html.parser')
    per_game_stats = player_soup.find(id='per_game_stats')
    adv_stats = player_soup.find(id='advanced')

    
    # Step 4: Convert the table into a DataFrame
    per_game_stats_df = pd.read_html(str(per_game_stats))[0]
    adv_stats_df = pd.read_html(str(adv_stats))[0]
    
    return player_name, per_game_stats_df, adv_stats_df

# --------------------------------------------------------------------------------------------------------
# TEAM STATS ---------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------

def extract_table_to_dataframe(data, table_id):
    """
    Extracts a table by its ID from an HTML file and converts it into a pandas DataFrame.
    Handles tables both directly in the HTML and within commented sections.

    Args:
        data (str): HTML data.
        table_id (str): ID of the table to extract.

    Returns:
        pd.DataFrame: DataFrame containing the table data, or None if the table is not found.
    """
    def find_table(soup, table_id):
        """Helper function to locate a table with a specific ID in a BeautifulSoup object."""
        return soup.find('table', id=table_id)

    def convert_table_to_dataframe(table):
        """Converts an HTML table into a pandas DataFrame."""
        # Extract headers
        headers = [th.get_text(strip=True) for th in table.find_all('th')]

        # Extract rows
        rows = []
        for row in table.find_all('tr'):
            cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
            if cells:  # Skip empty rows
                rows.append(cells)

        # Log headers and rows for debugging
        # print(f"Headers: {headers}")
        # print(f"Sample Row: {rows[0] if rows else 'No rows found'}")

        # Ensure consistent row lengths
        valid_rows = []
        for row in rows:
            if len(row) == len(headers):
                valid_rows.append(row)
            elif len(row) < len(headers):
                # Fill missing columns with None
                valid_rows.append(row + [None] * (len(headers) - len(row)))
            else:
                print(f"Skipping mismatched row: {row}")  # Log mismatched rows

        if not valid_rows:
            raise ValueError("No valid rows found to match headers.")

        # Create DataFrame
        return pd.DataFrame(valid_rows, columns=headers)

    def parse_html_or_comment(soup, table_id):
        """Tries to find and parse a table from HTML or comments."""
        # Look for the table directly in the HTML
        table = find_table(soup, table_id)
        if table:
            return convert_table_to_dataframe(table)

        # Search for the table in commented sections
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            commented_soup = BeautifulSoup(comment.strip(), 'html.parser')
            table = find_table(commented_soup, table_id)
            if table:
                return convert_table_to_dataframe(table)

        return None

    # Load and parse the HTML file
    # with open(file_path, 'r', encoding='utf-8') as file:
    #    soup = BeautifulSoup(data, 'html.parser')
    soup = BeautifulSoup(data, 'html.parser')

    # Extract the table as a DataFrame
    return parse_html_or_comment(soup, table_id)

def load_team_misc_and_opponent(season, team):
    team_url = f"https://www.basketball-reference.com/teams/{team}/{season}.html"

    data = requests.get(team_url).text

    _team_opps_df = extract_table_to_dataframe(data, "team_and_opponent")
    time.sleep(2)
    _misc_df = extract_table_to_dataframe(data, "team_misc")
    time.sleep(2)

    _team_opps_df.columns = _team_opps_df.iloc[0]
    _team_opps_df = _team_opps_df[1:].reset_index(drop=True)
    filtered_columns = [col for col in _team_opps_df.columns if col not in [None, '']]
    _team_opps_df = _team_opps_df[filtered_columns]

    _misc_df.columns = _misc_df.iloc[1]
    _misc_df = _misc_df[2:].reset_index(drop=True)
    filtered_columns = [col for col in _misc_df.columns if col not in [None, '']]
    _misc_df = _misc_df[filtered_columns]
    
    index_of_pace = list(_misc_df.columns).index('Pace')  # Dynamically find its position
    filtered_columns = _misc_df.columns[:index_of_pace + 1]
    _misc_df = _misc_df[filtered_columns]

    return _team_opps_df, _misc_df


# --------------------------------------------------------------------------------------------------------
# VISUALIZATION ------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------

    
def per_vs_team_success(player_df, team_df, mvp_df):
    mvp_list = list(mvp_df['Player'])
    # Merge player and team data on TEAM_ID and SEASON_ID
    merged_df = pd.merge(player_df, team_df, on=['TEAM_ID', 'SEASON_ID'], suffixes=('_player', '_team'))
    
    # Add a column to differentiate MVP candidates and the actual MVP
    actual_mvp = mvp_list[0]  # The first player in the list is the actual MVP
    merged_df['MVP_Candidate'] = merged_df['PLAYER_FULLNAME'].apply(
        lambda x: 'MVP' if x == actual_mvp else ('MVP Candidate' if x in mvp_list else 'No MVP votes')
    )
    
    # Scatterplot
    fig = px.scatter(
        merged_df, x='PER', y='W/L%',
        size='PTS_player', color='MVP_Candidate',
        hover_name='PLAYER_FULLNAME',
        title='Player Efficiency Rating (PER) vs Team Success (W/L%)',
        labels={'PER': 'Player Efficiency Rating (PER)', 'W/L%': 'Team Win/Loss Percentage'}
    )
    fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
    fig.update_layout(legend_title_text='Legend')
    return fig

def defense_vs_opponent_scoring(player_df, team_df, mvp_df):
    mvp_list = list(mvp_df['Player'])
    # Merge player and team data
    merged_df = pd.merge(player_df, team_df, on=['TEAM_ID', 'SEASON_ID'], suffixes=('_player', '_team'))
    
    # Add a column to differentiate MVP candidates and the actual MVP
    actual_mvp = mvp_list[0]
    merged_df['MVP_Candidate'] = merged_df['PLAYER_FULLNAME'].apply(
        lambda x: 'MVP' if x == actual_mvp else ('MVP Candidate' if x in mvp_list else 'No MVP votes')
    )
    
    # Scatterplot for defensive stats and opponent scoring
    fig = px.scatter(
        merged_df, x='BLK_player', y='PTS_opp_pg',
        size='STL_player', color='MVP_Candidate',
        hover_name='PLAYER_FULLNAME',
        title='Defensive Impact (BLK, STL) vs Opponent Points Per Game (PTS_opp_pg)',
        labels={'BLK_player': 'Player Blocks (BLK)', 'PTS_opp_pg': 'Opponent Points Per Game'}
    )
    fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
    fig.update_layout(legend_title_text='Legend')
    return fig

def scoring_vs_offensive_rating(player_df, team_df, mvp_df):
    """
    Creates a scatter plot highlighting MVP candidates and the actual MVP.

    Parameters:
    - player_df (DataFrame): The player statistics DataFrame.
    - team_df (DataFrame): The team statistics DataFrame.
    - mvp_list (list): List of player names who were MVP candidates, with the actual MVP as the first element.
    """
    mvp_list = list(mvp_df['Player'])
    # Merge player and team data
    merged_df = pd.merge(player_df, team_df, on=['TEAM_ID', 'SEASON_ID'], suffixes=('_player', '_team'))
    
    # Ensure non-negative values for bubble size
    merged_df['PER'] = merged_df['PER'].clip(lower=0)  # Replace negative PER with 0
    merged_df = merged_df.dropna(subset=['PER']) # Remove rows with NaN in PER
    
    # Add a column to differentiate MVP candidates and the actual MVP
    actual_mvp = mvp_list[0]  # The first player in the list is the actual MVP
    merged_df['MVP_Candidate'] = merged_df['PLAYER_FULLNAME'].apply(
        lambda x: 'MVP' if x == actual_mvp else ('MVP Candidate' if x in mvp_list else 'No MVP votes')
    )
    
    # Scatterplot
    fig = px.scatter(
        merged_df, x='PTS_player', y='ORtg',
        size='PER', color='MVP_Candidate',
        hover_name='PLAYER_FULLNAME',
        title='Scoring Impact (PTS) vs Team Offensive Rating (ORtg)',
        labels={'PTS_player': 'Player Points Scored (PTS)', 'ORtg': 'Team Offensive Rating (ORtg)'}
    )
    fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
    fig.update_layout(legend_title_text='Legend')
    return fig

def top_scorers_team_success(player_df, team_df, mvp_df):
    mvp_list = list(mvp_df['Player'])
    # Merge and filter top scorers
    merged_df = pd.merge(player_df, team_df, on=['TEAM_ID', 'SEASON_ID'], suffixes=('_player', '_team'))
    
    # Add a column to differentiate MVP candidates and the actual MVP
    actual_mvp = mvp_list[0]
    merged_df['MVP_Candidate'] = merged_df['PLAYER_FULLNAME'].apply(
        lambda x: 'MVP' if x == actual_mvp else ('MVP Candidate' if x in mvp_list else 'No MVP votes')
    )
    
    # Filter top 20 scorers
    top_scorers = merged_df.nlargest(20, 'PTS_player')  # Top 20 scorers
    
    # Bar chart
    fig = px.bar(
        top_scorers, x='PLAYER_FULLNAME', y='PTS_player',
        color='MVP_Candidate',
        title='Top Scorers and Team Success',
        labels={'PTS_player': 'Player Points Scored', 'W/L%': 'Team Win/Loss Percentage'},
        hover_name='TEAM_ABBREVIATION_player'
    )
    fig.update_xaxes(categoryorder='total descending')
    return fig

def advanced_metrics_player_contribution_filtered(player_df, team_df, mvp_df, dims=None, include_non_mvp=False, debug=False):
    """
    Creates an advanced metrics visualization highlighting MVP candidates with granularity based on 'Pts Won'.

    Parameters:
    - player_df (DataFrame): The player statistics DataFrame.
    - team_df (DataFrame): The team statistics DataFrame.
    - mvp_df (DataFrame): The MVP voting DataFrame.
    - include_non_mvp (bool): Whether to include non-MVP candidates in the visualization.
    """
    if not dims:
        dims = {
            'PLAYER_FULLNAME': 'Player',
            'eFG%_player': 'Effective FG %',
            'TS%_player' : 'True Shooting %',
            'PER': 'Player Efficiency (PER)',
            'WS': 'Win Shares',
            # 'WS/48': 'WS per 48 minutes (WS/48)',
            'W/L%': 'Win-Loss %',
            'Pace': 'Team Pace',
            'MOV': 'Margin of Victory (MOV)',
            'Pts_Won': 'MVP Points Won'
        }
    
    # Merge player and team data
    merged_df = pd.merge(player_df, team_df, on=['TEAM_ID', 'SEASON_ID'], suffixes=('_player', '_team'))

    # Add a column to differentiate MVP candidates
    merged_df['MVP_Candidate'] = merged_df['PLAYER_FULLNAME'].apply(
        lambda x: 'MVP Candidate' if x in mvp_df['Player'].values else 'Other'
    )

    # Merge MVP voting data into the player/team dataset
    merged_with_mvp = pd.merge(
        merged_df,
        mvp_df.drop(columns=['Age','Tm']),
        # mvp_df[['Player', 'Pts Won', 'Pts Max', 'Share', 
        #         'G', 'MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 
        #         'FG%', '3P%', 'FT%', 'WS', 'WS/48',
        #         'sentiment_1', 'sentiment_2', 'sentiment_3', 'sentiment_4',
        #         'sentiment_5', 'sentiment_6', 'sentiment_7', 'sentiment_8',
        #         'sentiment_9', 'sentiment_10', 'sentiment_11', 'sentiment_12',
        #         'sentiment_13', 'sentiment_14', 'sentiment_15']],  # Select key MVP metrics
        how='left',
        left_on='PLAYER_FULLNAME',
        right_on='Player'
    )

    # Replace metrics with MVP data where available, with fallback logic
    merged_with_mvp['WS'] = merged_with_mvp['WS_y'].fillna(merged_with_mvp['WS_x'])
    merged_with_mvp['PTS'] = merged_with_mvp['PTS'].fillna(merged_with_mvp['PTS_player'])
    merged_with_mvp['TRB'] = merged_with_mvp['TRB_y'].fillna(merged_with_mvp['REB'])
    merged_with_mvp['AST'] = merged_with_mvp.get('AST_y', merged_with_mvp['AST_player'])  # Fallback if 'AST_y' is missing

    # Optionally filter out non-MVP candidates
    if not include_non_mvp:
        merged_with_mvp = merged_with_mvp[merged_with_mvp['MVP_Candidate'] != 'Other']
    
    if debug:
        display(sorted(list(merged_with_mvp.columns)))
    
    # Drop unnecessary columns and avoid confusion between suffixes
    merged_with_mvp = merged_with_mvp.rename(columns={'Pts Won': 'Pts_Won'})
    analysis_df = merged_with_mvp[list(dims.keys())]

    # Parallel coordinates plot for multidimensional analysis
    fig = px.parallel_coordinates(
        analysis_df,
        dimensions=list(dims.keys()),  # Include player name as the first dimension
        color='Pts_Won',
        color_continuous_scale='Viridis',  # Granularity of 'Pts Won'
        labels=dims,
        title='Advanced Metrics: Player Contribution Highlighting MVP Points'
    )

    # Rotate labels by setting tickangle for each axis
    fig.update_xaxes(tickangle=-45)
        
    return fig

def advanced_metrics_scatter_matrix(player_df, team_df, mvp_df, include_non_mvp=False):
    mvp_list = list(mvp_df['Player'])
    # Merge player and team data
    merged_df = pd.merge(player_df, team_df, on=['TEAM_ID', 'SEASON_ID'], suffixes=('_player', '_team'))
    
    # Add a column to differentiate MVP candidates and the actual MVP
    actual_mvp = mvp_list[0]
    merged_df['MVP_Candidate'] = merged_df['PLAYER_FULLNAME'].apply(
        lambda x: actual_mvp if x == actual_mvp else ('MVP Candidate' if x in mvp_list else 'Other')
    )

    # Optionally filter out non-MVP candidates
    if not include_non_mvp:
        merged_df = merged_df[merged_df['MVP_Candidate'] != 'Other']
    
    # Scatter matrix
    fig = px.scatter_matrix(
        merged_df,
        dimensions=['PER', 'WS', 'Pace', 'MOV'],  # Metrics to include
        color='MVP_Candidate',
        title='Scatter Matrix: Advanced Metrics and MVP Highlights',
        labels={
            'PER': 'PER',
            'WS': 'WS',
            'Pace': 'Team Pace',
            'MOV': 'MOV'
        }
    )
    fig.update_traces(diagonal_visible=False)  # Hide diagonal density plots
    return fig

def per_ws_vs_mvp_points(player_df, team_df, mvp_df):
    """
    Visualizes the relationship between PER * Win Shares and MVP points won.

    Parameters:
    - player_df (DataFrame): The player statistics DataFrame.
    - team_df (DataFrame): The team statistics DataFrame.
    - mvp_df (DataFrame): The MVP voting DataFrame.

    Returns:
    - fig: A Plotly scatter plot figure.
    """
    import pandas as pd
    import plotly.express as px

    # Merge player and team data
    merged_df = pd.merge(player_df, team_df, on=['TEAM_ID', 'SEASON_ID'], suffixes=('_player', '_team'))

    # Add MVP data
    merged_with_mvp = pd.merge(
        merged_df,
        mvp_df[['Player', 'Pts Won', 'WS']],  # Use only relevant MVP columns
        how='inner',  # Only keep players with MVP votes
        left_on='PLAYER_FULLNAME',
        right_on='Player',
        suffixes=('', '_mvp')
    )

    # Calculate PER * WS
    merged_with_mvp['PER * WS'] = merged_with_mvp['PER'] * merged_with_mvp['WS_mvp']

    # Scatterplot
    fig = px.scatter(
        merged_with_mvp,
        x='PER * WS',
        y='Pts Won',
        size='BPM',  # Use PER * WS for bubble size
        color='PLAYER_FULLNAME',  # Color by player
        hover_name='PLAYER_FULLNAME',
        title='Impact of PER * Win Shares on MVP Points',
        labels={
            'PER * WS': 'PER * Win Shares',
            'Pts Won': 'MVP Points Won',
            'PTS_player': 'Points Scored (PTS)'
        }
    )
    fig.update_traces(marker=dict(opacity=0.8))
    return fig

### MVP DATA ONLY

def voting_share_vs_points(df):
    fig = px.scatter(
        df, x='Share', y='PTS',
        size='WS', color='Tm',
        hover_name='Player',
        title='MVP Voting Share vs. Points Per Game',
        labels={'Share': 'MVP Voting Share', 'PTS': 'Points Per Game', 'WS': 'Win Shares'}
    )
    return fig

# Example usage

def voting_share_distribution(df):
    team_shares = df.groupby('Tm')['Share'].sum().reset_index()
    
    fig = px.bar(
        team_shares, x='Tm', y='Share',
        title='Distribution of MVP Votes Among Teams',
        labels={'Tm': 'Team', 'Share': 'Total MVP Voting Share'},
        text='Share'
    )
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    return fig

def voting_share_vs_ws_per_48(df):
    fig = px.scatter(
        df, x='Share', y='WS/48',
        size='PTS', color='Tm',
        hover_name='Player',
        title='MVP Voting Share vs. WS/48',
        labels={'Share': 'MVP Voting Share', 'WS/48': 'Win Shares Per 48 Minutes', 'PTS': 'Points Per Game'}
    )
    return fig

def player_performance_parallel(df):
    fig = px.parallel_coordinates(
        df,
        dimensions=['PTS', 'TRB', 'AST', 'WS', 'WS/48'],
        color='Share',
        title='Player Performance Metrics of MVP Candidates',
        labels={'PTS': 'Points Per Game', 'TRB': 'Rebounds Per Game', 'AST': 'Assists Per Game',
                'WS': 'Win Shares', 'WS/48': 'Win Shares Per 48 Minutes', 'Share': 'MVP Voting Share'}
    )
    return fig

def age_vs_voting_share(df):
    fig = px.scatter(
        df, x='Age', y='Share',
        size='PTS', color='Tm',
        hover_name='Player',
        title='Age vs. MVP Voting Share',
        labels={'Age': 'Player Age', 'Share': 'MVP Voting Share', 'PTS': 'Points Per Game'}
    )
    return fig

def team_representation(df):
    team_counts = df['Tm'].value_counts().reset_index()
    team_counts.columns = ['Team', 'Count']
    
    fig = px.bar(
        team_counts, x='Team', y='Count',
        title='Team Representation in MVP Voting',
        labels={'Team': 'Team', 'Count': 'Number of MVP Candidates'},
        text='Count'
    )
    fig.update_traces(textposition='outside')
    return fig

# --------------------------------------------------------------------------------------------------------
# SENTIMENT ----------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------

def extract_rating(input_string):
    """
    Extracts the first single digit in the format [[X / Y]] from the input string.
    
    Args:
    input_string (str): The input string containing the rating.
    
    Returns:
    int: The extracted single digit as an integer.
    """
    match = re.search(r'\[\s*\[\s*(\-*\d+\.*\d*)\s*/\s*\d+\s*\]\s*\]', input_string)
    if match:
        return float(match.group(1))
    else:
        print(input_string)
        print("Rating in the format [[X / Y]] not found in the input string.")
        return -1

def add_sentiment_avg(df):
    """
    Adds a column `sentiment_avg` to the DataFrame, which is the average
    of numeric data from the specified columns for each row.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: Updated DataFrame with the new `sentiment_avg` column
    """

    sentiment_cols = [
    'sentiment_1', 'sentiment_2', 'sentiment_3', 'sentiment_4',
    'sentiment_5', 'sentiment_6', 'sentiment_7', 'sentiment_8',
    'sentiment_9', 'sentiment_10', 'sentiment_11', 'sentiment_12',
    'sentiment_13', 'sentiment_14', 'sentiment_15'
]
    df["sentiment_avg"] = df[sentiment_cols].mean(axis=1)
    return df


def analyze_sentiment(client, model_name, role, prompt, year, player_name, temperature=1.0, top_p=1.0, max_tokens=1000):
    content = f'Consider the {year}-{str(year+1)[2:]} NBA season.  Give your response with respect to {player_name}. {prompt['prompt']}.'

    
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": role,
                },
                {
                    "role": "user",
                    "content": content,
                }
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            model=model_name
        )
    except Exception as e:
        print (f"Failed to receive response from OpenAI client with error: {str(e)}")    
        
    text = response.choices[0].message.content
    append_output_to_file(f'{write_season(year)} {player_name}\n{prompt['title']}: {text}')
    rating = extract_rating(text)

    print(f'{write_season(year)}: {player_name} - {prompt['title']}: {rating}')
        
    return rating
        
def tell_mvp_story(client, model_name, role, prompts, year, player_name, temperature=1.0, top_p=1.0, max_tokens=1000, sleep=7):
    ratings = {
        player_name:{}
    }
    
    for prompt in prompts.keys():
        rating = analyze_sentiment(
            client=client,                     
            model_name=model_name,                     
            role=role,                     
            prompt=prompts[prompt],   
            year=year,
            player_name=player_name,                     
            temperature=temperature,                     
            top_p=top_p,                     
            max_tokens=max_tokens
        )

        ratings[player_name][prompt] = rating

        time.sleep(sleep) # Accomodate request limit ~10/min

    return ratings

def process_mvp_stories_for_year(client, model_name, role, prompts, year, df, temperature=1.0, top_p=1.0, max_tokens=1000, sleep=7):
    """
    Processes the MVP story for the first five players in the DataFrame and
    writes the sentiment ratings into new columns for each prompt key.

    Parameters:
    - client (object): ChatGPT client instance.
    - model_name (str): The model name to use for predictions.
    - role (str): System role for ChatGPT.
    - prompts (dict): Dictionary of prompts.
    - year (int): NBA season year (e.g., 2023 for the 2023-24 season).
    - temperature (float): Temperature parameter for ChatGPT.
    - top_p (float): Top-p parameter for ChatGPT.
    - max_tokens (int): Maximum tokens for each ChatGPT response.
    - sleep (int): Sleep duration between requests to avoid rate limits.

    Saves:
    - Overwrites the CSV with new sentiment columns added.
    """

    # Initialize new columns for each prompt key
    for prompt_key in prompts.keys():
        df[f"sentiment_{prompt_key}"] = 0  # Default to 0 for all rows

    # Process only the top 5 players based on the 'Rank' column
    for index, row in df.iterrows():
        if index < 7:  # Only process the first 7 rows
            player_name = row['Player']
            ratings = tell_mvp_story(
                client=client,
                model_name=model_name,
                role=role,
                prompts=prompts,
                year=year,
                player_name=player_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                sleep=sleep
            )

            # Add ratings for each prompt key to the corresponding column
            for prompt_key, rating in ratings[player_name].items():
                df.at[index, f"sentiment_{prompt_key}"] = rating

            df = add_sentiment_avg(df)
        else:
            # Skip processing for rows beyond the top 5
            break

    # Save the modified DataFrame back to the file
    file_path = f'/Users/cb/src/nba_mvp_ml/data/processed/by_season/mvp/sentiment/mvp_{year}-{str(year+1)[2:]}.csv'
    df.to_csv(file_path, index=False)
    print(f"Updated DataFrame saved to {file_path}")



# --------------------------------------------------------------------------------------------------------
# MACHINE LEARNING ---------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------

# Function to load and preprocess data
def load_and_preprocess_data(file_path, target_column='mvp', remove_excess_features=False):
    excess_features = ['STL_opp_pg', 'ASTPct', 'DREB', 'PF_opp', 'FTA_team', 'sentiment_4', 'PF_pg', 'DREB_PG', 'PTS_player', 'sentiment_9', 
                       'TOV_player', 'OWS', 'sentiment_7', 'FGM_PG', 'FT_PCT_PG', 'STL_opp', 'FG%_opp', 'AST_opp_pg', 'FGA_player', 'FG3A', 
                       'TS%_player', 'FTA_player', 'TOV_pg', 'REB', 'NRtg', 'FTM', 'FT%_y', 'AST_pg', 'STLPct', 'DRB%', 'FTr_player', 
                       'PF_opp_pg', 'FT_PCT', 'eFG%_player', 'FG_pg', 'PTS_PG', 'FT%_pg', 'index', 'FT_opp', 'ORB', 'DRB_opp', 'TRB_opp', 
                       'W', 'FGA_PG', '3P%_x', '3P%_pg', '3PA_opp_pg', 'eFG%.1', 'TSPct', 'FTA_opp_pg', 'DRB', 'BLK', '2PA_pg', 'TOV%.1', 
                       '2P%_opp', '2PA', 'STL', 'G_y', 'FTr_team', 'PA/G', 'BLK_team', 'MIN_PG', 'FG_PCT', '2PA_opp', 'eFG%_team', 'DBPM', 
                       '3P%_opp_pg', 'TRB_opp_pg', 'FGA_pg', 'TRB_y', 'USGPct', 'AST', 'ORB_pg', 'AST_player', 'Pace', 'FT', 'MP_pg', '3PA_opp', 
                       '2P', 'PTS', 'FT%_x', 'sentiment_11', 'AST_opp', 'SRS_wl', '3P_opp', 'Rk_pg', 'PTS_team', '3PAr_player', 'W/L%', 'L', 
                       'AST_PG', 'FG%_opp_pg', '3P%_opp', 'TOV_opp_pg', 'FTM_PG', 'BLK_player', 'sentiment_10', 'FG_PCT_PG', 'BLKPct', 
                       'FG_opp', 'PW', 'ORB_opp', 'GS', 'ORB_opp_pg', 'PS/G', 'sentiment_15', '2P_pg', 'TOV_PG', 'BLK_opp_pg', 'FTA_pg', 
                       'DRB_pg', 'FT/FGA.1', 'TRB', 'PF_PG', 'PTS_opp', 'ORB%', 'STL_player', 'FT_pg', 'PF_player', 'FG3_PCT_PG', 'PF_team', 
                       '2P_opp', 'FTA_PG', 'MIN', 'SOS', 'TRB_x', 'FT%_opp', 'ORBPct', 'TRBPct', 'FG3M', 'PL', 'FGA_opp', 'PTS_opp_pg', 'GP', 
                       'BLK_pg', 'STL_team', 'MP_opp', 'Rk_opp', 'FT_opp_pg', '3PA', 'REB_PG', 'AST_team', 'PTS_pg', 'TS%_team', 'STL_PG', 'OREB',
                       'FG3A_PG', 'FGA_opp_pg', 'TOV_team', '2P%_pg', 'TRB_pg', 'L_wl', 'MP_opp_pg', '3PAr_team', '3P_opp_pg', 'FG_opp_pg', 'TOV%', 
                       'sentiment_12', '2PA_opp_pg', 'TOV_opp', 'W_wl', 'FG%_y', 'FG%_x', 'OREB_PG', 'MOV', 'FGA_team', 'DRtg', '2P%', 
                       'PLAYER_AGE', 'BLK_opp', 'FTA_opp', 'G_pg', 'FT%_opp_pg', 'FG3M_PG', '3P', 'STL_pg', 'FT/FGA', 'FG%_pg', '2P_opp_pg', 
                       'MP_x', 'G_opp_pg', '3PA_pg', '3P_pg', 'G_x', 'BLK_PG', 'G_opp', 'FG']
    
    repeat_cols = ['2P%', '2P%_opp', '3P%_opp', '3P%_x', '3P%_y', 'FG3_PCT', 'AST_player', 'BLK', 'FG%_opp', 'FG%_x', 'FG%_y', 'FG_PCT', 
                   'FT%_opp', 'FT%_x', 'FT%_y', 'FT_PCT', 'G_y', 'G_opp', 'G_pg', 'G_x', 'MP_y', 'MP_x', 'MP_pg', 'PA/G', 
                   'PS/G', 'PTS', 'TRB', 'TRB_y', 'SRS_wl', 'STL', 'TRB_y', 'WS_x', 'WS_y', 'WS/48_y']
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Separate features and target
    if remove_excess_features:
        X = df.drop(columns=['index'] + [target_column] + excess_features + repeat_cols)
    else:
        X = df.drop(columns=[target_column] + repeat_cols)
    
    y = df[target_column]
    
    # Normalize features
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")  # Set the output to be a Pandas DataFrame
    X_normalized = scaler.fit_transform(X)
    
    return X_normalized, y

# Function to split data
def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    # Split data into train and temp (test + validation)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=random_state)
    
    # Split temp into test and validation
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

