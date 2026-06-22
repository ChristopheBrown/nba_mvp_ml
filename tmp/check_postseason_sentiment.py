from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.pipeline import _sentiment_for_player

for name in [
    "Nikola Jokic",
    "Luka Doncic",
    "Shai Gilgeous-Alexander",
    "Victor Wembanyama",
]:
    vals, avg = _sentiment_for_player(name, {})
    print(name, avg, vals['sentiment_1'], vals['sentiment_2'], vals['sentiment_3'], vals['sentiment_14'])
