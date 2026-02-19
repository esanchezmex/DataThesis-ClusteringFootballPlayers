"""
Player-to-team mapping utilities for tracking data.
"""

from pathlib import Path


def get_player_team_mapping(file_path, players_df):
    """
    Extract match_id from file path and create player_id -> team_id mapping.

    Parameters
    ----------
    file_path : str
        Path to the tracking JSON file (e.g., 'tracking_usl_championship-2025-2006551.json')
    players_df : pandas.DataFrame
        DataFrame with columns: ['match_id', 'player_id', 'team_id', ...]

    Returns
    -------
    tuple
        (player_to_team, home_team_id, away_team_id, match_id)
        - player_to_team: dict mapping player_id (str) to team_id (str)
        - home_team_id: Team ID for home team (blue)
        - away_team_id: Team ID for away team (red)
        - match_id: The extracted match ID
    """
    # Extract match_id from filename
    match_id = int(Path(file_path).stem.split("-")[-1])

    # Try both string and int versions of match_id
    match_players_str = players_df[players_df["match_id"] == str(match_id)].copy()
    match_players_int = players_df[players_df["match_id"] == match_id].copy()

    # Use whichever works
    if len(match_players_str) > 0:
        match_players = match_players_str
    elif len(match_players_int) > 0:
        match_players = match_players_int
    else:
        print(f"Warning: No players found for match_id {match_id}")
        return {}, None, None, match_id

    # Create player_id -> team_id mapping (convert to strings for consistency)
    player_to_team = dict(
        zip(
            match_players["player_id"].astype(str),
            match_players["team_id"].astype(str),
        )
    )

    # Get unique team_ids and assign home/away
    unique_teams = sorted(match_players["team_id"].unique())
    home_team_id = unique_teams[0] if len(unique_teams) > 0 else None
    away_team_id = unique_teams[1] if len(unique_teams) > 1 else None

    return player_to_team, home_team_id, away_team_id, match_id


def split_players_by_team(player_data, player_to_team_map, home_team_id, away_team_id):
    """
    Split players into home and away teams based on their team_id.

    Parameters
    ----------
    player_data : list
        List of player dictionaries from tracking data frame
    player_to_team_map : dict
        Mapping from player_id (str) to team_id (str)
    home_team_id : str
        Team ID for home team (will be colored blue)
    away_team_id : str
        Team ID for away team (will be colored red)

    Returns
    -------
    tuple
        ((away_x, away_y), (home_x, home_y)) - Coordinates for away and home players
    """
    home_x, home_y = [], []
    away_x, away_y = [], []

    # If no mapping available, fall back to x-coordinate split
    if not player_to_team_map or home_team_id is None or away_team_id is None:
        for p in player_data:
            x = p.get("x")
            y = p.get("y")
            if x is None or y is None:
                continue
            if x < 0:
                away_x.append(x)
                away_y.append(y)
            else:
                home_x.append(x)
                home_y.append(y)
        return (away_x, away_y), (home_x, home_y)

    # Use team mapping
    for p in player_data:
        player_id = p.get("player_id")
        x = p.get("x")
        y = p.get("y")

        if player_id is None or x is None or y is None:
            continue

        # Convert player_id to string for lookup
        team_id = player_to_team_map.get(str(player_id))

        if team_id == str(home_team_id):
            home_x.append(x)
            home_y.append(y)
        elif team_id == str(away_team_id):
            away_x.append(x)
            away_y.append(y)
        # If team_id not found, skip (could be referee, etc.)

    return (away_x, away_y), (home_x, home_y)
