from pathlib import Path


class ProjectConfig:
    # Get the project root directory
    ROOT_DIR = Path(__file__).parent.parent.parent
    print(f"Root directory is: {ROOT_DIR}")

    # Data paths
    DATA_DIR = ROOT_DIR / "data"
    print(f"Data directory is: {DATA_DIR}")
    RAW_DATA_DIR = DATA_DIR / "raw"
    print(f"Raw data directory is: {RAW_DATA_DIR}")
    PROCESSED_DATA_DIR = DATA_DIR / "processed"

    # File paths
    RAW_DATA_FILE = RAW_DATA_DIR / "all_seasons.csv"
    print(f"Looking for file at: {RAW_DATA_FILE}")

    # Data processing parameters
    NUMERICAL_COLUMNS = [
        "age",
        "player_height",
        "player_weight",
        "gp",
        "pts",
        "reb",
        "ast",
        "net_rating",
        "oreb_pct",
        "dreb_pct",
        "usg_pct",
        "ts_pct",
        "ast_pct",
    ]

    CATEGORICAL_COLUMNS = [
        "team_abbreviation",
        "college",
        "country",
        "draft_year",
        "draft_round",
        "draft_number",
    ]
