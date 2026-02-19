"""
Football pitch visualization.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np


def create_pitch(
    ax,
    x_min=-60.1,
    x_max=60.3,
    y_min=-41.7,
    y_max=39.2,
    pitch_length=105,
    pitch_width=68,
    title="Football Match",
    attacking_direction="right",
):
    """
    Create a football pitch visualization on the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the pitch on
    x_min, x_max, y_min, y_max : float
        Plot limits for the axes
    pitch_length, pitch_width : float
        Dimensions of the pitch in meters
    title : str
        Title for the plot
    attacking_direction : str
        Direction of attack arrow: "right" (default) or "left"

    Returns
    -------
    matplotlib.axes.Axes
        The configured axes
    """
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_facecolor("#3f995b")  # Grass green
    ax.set_title(title, fontsize=16, color="white", weight="bold")

    # Standard football dimensions
    goal_width = 7.32  # meters
    goal_area_depth = 5.5  # 6-yard box depth
    goal_area_width = 18.32  # 6-yard box width
    penalty_area_depth = 16.5  # 18-yard box depth
    penalty_area_width = 40.32  # 18-yard box width
    penalty_spot_distance = 11.0  # Distance from goal line to penalty spot

    # Draw pitch outline
    ax.plot(
        [
            -pitch_length / 2,
            pitch_length / 2,
            pitch_length / 2,
            -pitch_length / 2,
            -pitch_length / 2,
        ],
        [
            -pitch_width / 2,
            -pitch_width / 2,
            pitch_width / 2,
            pitch_width / 2,
            -pitch_width / 2,
        ],
        color="white",
        linewidth=2.5,
    )

    # Draw center line
    ax.axvline(0, color="white", linewidth=1.5, linestyle="--", alpha=0.7)

    # Draw center circle
    center_circle = plt.Circle(
        (0, 0), 9.15, fill=False, color="white", linewidth=1.5, linestyle="--", alpha=0.7
    )
    ax.add_patch(center_circle)

    # Draw center spot
    ax.plot(0, 0, "o", color="white", markersize=4, zorder=5)

    # Left goal (x = -pitch_length/2)
    left_goal_x = -pitch_length / 2
    # Goal posts
    ax.plot(
        [left_goal_x, left_goal_x],
        [-goal_width / 2, goal_width / 2],
        color="white",
        linewidth=3,
        zorder=5,
    )
    # Goal net (visual representation)
    goal_net = Rectangle(
        (left_goal_x - 0.5, -goal_width / 2),
        0.5,
        goal_width,
        fill=True,
        facecolor="white",
        alpha=0.3,
        edgecolor="white",
        linewidth=1,
        zorder=4,
    )
    ax.add_patch(goal_net)

    # Left goal area (6-yard box)
    left_goal_area = Rectangle(
        (left_goal_x, -goal_area_width / 2),
        goal_area_depth,
        goal_area_width,
        fill=False,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(left_goal_area)

    # Left penalty area (18-yard box)
    left_penalty_area = Rectangle(
        (left_goal_x, -penalty_area_width / 2),
        penalty_area_depth,
        penalty_area_width,
        fill=False,
        edgecolor="white",
        linewidth=2,
        zorder=3,
    )
    ax.add_patch(left_penalty_area)

    # Left penalty spot
    ax.plot(
        left_goal_x + penalty_spot_distance,
        0,
        "o",
        color="white",
        markersize=6,
        zorder=5,
    )

    # Right goal (x = pitch_length/2)
    right_goal_x = pitch_length / 2
    # Goal posts
    ax.plot(
        [right_goal_x, right_goal_x],
        [-goal_width / 2, goal_width / 2],
        color="white",
        linewidth=3,
        zorder=5,
    )
    # Goal net (visual representation)
    goal_net_right = Rectangle(
        (right_goal_x, -goal_width / 2),
        0.5,
        goal_width,
        fill=True,
        facecolor="white",
        alpha=0.3,
        edgecolor="white",
        linewidth=1,
        zorder=4,
    )
    ax.add_patch(goal_net_right)

    # Right goal area (6-yard box)
    right_goal_area = Rectangle(
        (right_goal_x - goal_area_depth, -goal_area_width / 2),
        goal_area_depth,
        goal_area_width,
        fill=False,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(right_goal_area)

    # Right penalty area (18-yard box)
    right_penalty_area = Rectangle(
        (right_goal_x - penalty_area_depth, -penalty_area_width / 2),
        penalty_area_depth,
        penalty_area_width,
        fill=False,
        edgecolor="white",
        linewidth=2,
        zorder=3,
    )
    ax.add_patch(right_penalty_area)

    # Right penalty spot
    ax.plot(
        right_goal_x - penalty_spot_distance,
        0,
        "o",
        color="white",
        markersize=6,
        zorder=5,
    )

    # Draw attacking direction arrow (at bottom of pitch)
    arrow_length = 8
    arrow_y = -pitch_width / 2 - 2

    if attacking_direction == "right":
        arrow = FancyArrowPatch(
            (x_min + 2, arrow_y),
            (x_min + 2 + arrow_length, arrow_y),
            arrowstyle="->",
            mutation_scale=20,
            color="yellow",
            linewidth=2.5,
            zorder=10,
        )
        ax.text(
            x_min + 2 + arrow_length / 2,
            arrow_y - 1.5,
            "Attacking",
            ha="center",
            color="yellow",
            fontsize=10,
            weight="bold",
            zorder=10,
        )
    else:  # left
        arrow = FancyArrowPatch(
            (x_max - 2, arrow_y),
            (x_max - 2 - arrow_length, arrow_y),
            arrowstyle="->",
            mutation_scale=20,
            color="yellow",
            linewidth=2.5,
            zorder=10,
        )
        ax.text(
            x_max - 2 - arrow_length / 2,
            arrow_y - 1.5,
            "Attacking",
            ha="center",
            color="yellow",
            fontsize=10,
            weight="bold",
            zorder=10,
        )

    ax.add_patch(arrow)

    return ax
