import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from nba_analysis.data_processor import DataProcessor
from nba_analysis.utils import get_player_career_stats

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def main():
    # Initialize and process data
    processor = DataProcessor()
    df = (
        processor.load_data()
        .clean_data()
        .add_derived_features()
        .filter_data(min_games=10)
        .get_processed_data()
    )

    # Analysis section
    print("Data Overview:")
    print("-" * 50)
    print(f"Total number of records: {len(df)}")
    print(f"Unique players: {df['player_name'].nunique()}")
    print(f"Seasons covered: {df['season'].nunique()}")

    # Player analysis (just choosing 1 player for simplicity)
    player_name = "LeBron James"
    stats = get_player_career_stats(df, player_name)
    print(f"\n{player_name}'s Career Stats:")
    print("-" * 50)
    for stat, value in stats.items():
        print(f"{stat}: {value:.2f}")

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="age", y="pts", alpha=0.5)
    plt.title("Age vs Points per Game")
    plt.savefig(project_root / "data" / "processed" / "age_vs_points.png")
    plt.close()


if __name__ == "__main__":
    main()
