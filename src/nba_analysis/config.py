from pathlib import Path


class ProjectConfig:
    # Get the project root directory
    ROOT_DIR = Path(__file__).parent.parent

    # Data paths
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"

    # File paths
    RAW_DATA_FILE = RAW_DATA_DIR / "all_seasons.csv"

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
