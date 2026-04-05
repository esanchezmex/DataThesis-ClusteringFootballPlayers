"""Shared paths relative to the repository root."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PROCESSED_PLAYER_PROFILES_PKL = REPO_ROOT / "processed_player_profiles.pkl"
