"""
Download the fixed SoccerNet half this experiment runs on (Manchester City vs
Barcelona, UCL 2016-11-01, half 2, 720p) into SOCCERNET_EXPERIMENT_DIR.

SoccerNet's downloader places the file at
  <SOCCERNET_EXPERIMENT_DIR>/<GAME path>/2_720p.mkv
and the extractor's rglob picks it up from there. No moving / renaming.

Needs SoccerNetv2_password in tackle-detection-model-comparison/.env. Run on the login node:
  uv run python soccernet_experiment/download_half.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

THESIS_CODE = Path(__file__).resolve().parent.parent
load_dotenv(THESIS_CODE / ".env")
sys.path.insert(0, str(THESIS_CODE / "src"))

from SoccerNet.Downloader import SoccerNetDownloader

from config import SOCCERNET_EXPERIMENT_DIR

GAME = ("europe_uefa-champions-league/2016-2017/"
        "2016-11-01 - 22-45 Manchester City 3 - 1 Barcelona")
VIDEO_FILE = "2_720p.mkv"   # half 2, HQ


def main():
    password = os.getenv("SoccerNetv2_password")
    downloader = SoccerNetDownloader(LocalDirectory=str(SOCCERNET_EXPERIMENT_DIR))
    downloader.password = password
    downloader.downloadGame(game=GAME, files=[VIDEO_FILE])


if __name__ == "__main__":
    main()
