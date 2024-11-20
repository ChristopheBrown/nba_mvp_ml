from nba_api.stats.endpoints import playercareerstats
from nba_api.live.nba.endpoints import scoreboard


def test_PlayerCareerStats():
    career = playercareerstats.PlayerCareerStats(player_id='203999') 
    assert career != None

def test_ScoreBoard():
    games = scoreboard.ScoreBoard()
    assert games != None