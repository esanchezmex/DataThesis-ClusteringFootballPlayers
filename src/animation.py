"""
Tracking data animation utilities.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .pitch import create_pitch
from .player_mapping import get_player_team_mapping, split_players_by_team


def create_tracking_animation(
    file_path,
    players_df,
    start_frame=0,
    end_frame=100,
    save_path=None,
    fps=20,
    interval=20,
):
    """
    Create an animation from tracking data.

    Parameters
    ----------
    file_path : str
        Path to tracking JSON file
    players_df : pandas.DataFrame
        DataFrame with player-team mappings
    start_frame, end_frame : int
        Frame range to animate
    save_path : str or None
        If provided, save animation to this path
    fps : int
        Frames per second for saved video
    interval : int
        Milliseconds between frames in animation

    Returns
    -------
    tuple
        (fig, anim) - matplotlib Figure and FuncAnimation objects
    """
    # Load tracking data
    with open(file_path, "r") as f:
        frames = json.load(f)

    frames_slice = frames[start_frame : end_frame + 1]

    # Get player-team mapping
    player_to_team, home_team_id, away_team_id, match_id = get_player_team_mapping(
        file_path, players_df
    )

    # Create figure and pitch
    fig, ax = plt.subplots(figsize=(14, 10))
    create_pitch(ax, title=f"Match {match_id} – Frames {start_frame}–{end_frame}")

    # Initialize scatter plots
    away_scatter = ax.scatter(
        [], [],
        c="red",
        s=80,
        label="Away",
        edgecolors="white",
        linewidths=1,
        zorder=5,
        alpha=0.9,
    )
    home_scatter = ax.scatter(
        [], [],
        c="blue",
        s=80,
        label="Home",
        edgecolors="white",
        linewidths=1,
        zorder=5,
        alpha=0.9,
    )
    ball_scatter = ax.scatter(
        [], [],
        c="black",
        s=100,
        label="Ball",
        edgecolors="white",
        linewidths=1.5,
        zorder=10,
        alpha=0.95,
    )

    timestamp_text = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        color="white",
        fontsize=13,
        weight="bold",
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="black",
            alpha=0.8,
            edgecolor="white",
            linewidth=1.5,
        ),
    )

    ax.legend(
        loc="upper right",
        facecolor="white",
        framealpha=0.95,
        fontsize=11,
        edgecolor="black",
        frameon=True,
    )

    # Animation update function
    def update(i):
        frame = frames_slice[i]
        player_data = frame.get("player_data", [])
        (away_x, away_y), (home_x, home_y) = split_players_by_team(
            player_data, player_to_team, home_team_id, away_team_id
        )

        # Update scatter plots
        if away_x:
            away_scatter.set_offsets(np.column_stack([away_x, away_y]))
        else:
            away_scatter.set_offsets(np.empty((0, 2)))

        if home_x:
            home_scatter.set_offsets(np.column_stack([home_x, home_y]))
        else:
            home_scatter.set_offsets(np.empty((0, 2)))

        # Update ball
        ball = frame.get("ball_data", {})
        if (
            ball.get("is_detected")
            and ball.get("x") is not None
            and ball.get("y") is not None
        ):
            ball_scatter.set_offsets(np.array([[ball["x"], ball["y"]]], dtype=float))
            ball_scatter.set_visible(True)
        else:
            ball_scatter.set_offsets(np.empty((0, 2)))
            ball_scatter.set_visible(False)

        # Update timestamp
        ts = frame.get("timestamp", "N/A")
        period = frame.get("period", "N/A")
        frame_num = start_frame + i
        timestamp_text.set_text(
            f"Frame: {frame_num} | Period: {period} | Time: {ts}\n"
            f"Away: {len(away_x)} players | Home: {len(home_x)} players"
        )

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames_slice),
        interval=interval,
        repeat=True,
        blit=False,
    )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer="ffmpeg", fps=fps, bitrate=1800)
        print("Animation saved!")

    return fig, anim
