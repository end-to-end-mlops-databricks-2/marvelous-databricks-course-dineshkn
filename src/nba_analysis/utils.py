def calculate_efficiency(row):
    """Calculate player efficiency based on basic stats."""
    return (row["pts"] + row["reb"] + row["ast"]) / row["gp"] if row["gp"] > 0 else 0


def get_player_career_stats(data, player_name):
    """Get career statistics for a specific player."""
    player_data = data[data["player_name"] == player_name]

    career_stats = {
        "seasons_played": len(player_data),
        "avg_points": player_data["pts"].mean(),
        "avg_rebounds": player_data["reb"].mean(),
        "avg_assists": player_data["ast"].mean(),
        "total_games": player_data["gp"].sum(),
    }

    return career_stats
