import os
import sys
from pathlib import Path

import pandas as pd
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from nba_analysis.config import ProjectConfig

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

print(f"Looking for file at: {ProjectConfig.RAW_DATA_FILE}")


class DataProcessor:
    def __init__(self):
        self.config = ProjectConfig()
        self.data = None

    def load_data(self):
        """Load the raw NBA players data."""
        file_path = self.config.RAW_DATA_FILE.resolve()

        print(f"Current working directory: {Path.cwd()}")
        print(f"File path: {file_path}")
        print(f"File exists: {file_path.exists()}")
        print(f"Is file readable: {os.access(file_path, os.R_OK)}")

        self.data = pd.read_csv(file_path)
        return self

    def clean_data(self):
        """Perform basic data cleaning operations."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Drop the unnamed index column
        if "Unnamed: 0" in self.data.columns:
            self.data = self.data.drop("Unnamed: 0", axis=1)

        # Handle missing values
        self.data = self.data.fillna(
            {
                "college": "Unknown",
                "country": "Unknown",
                "draft_year": "Undrafted",
                "draft_round": "Undrafted",
                "draft_number": "Undrafted",
            }
        )

        return self

    def add_derived_features(self):
        """Add calculated features to the dataset."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Calculate BMI
        self.data["bmi"] = self.data["player_weight"] / (
            self.data["player_height"] ** 2
        )

        # Calculate points per minute (assuming games are 48 minutes)
        self.data["points_per_minute"] = self.data["pts"] / (self.data["gp"] * 48)

        # Calculate overall efficiency
        self.data["efficiency"] = (
            self.data["pts"]
            + self.data["reb"]
            + self.data["ast"]
            - (self.data["player_weight"] - 80)
            * 0.1  # Small penalty for being too heavy
        )

        return self

    def filter_data(self, min_games=20, min_points=5):
        """Filter data based on minimum criteria."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.data = self.data[
            (self.data["gp"] >= min_games) & (self.data["pts"] >= min_points)
        ]
        return self

    def get_processed_data(self):
        """Return the processed dataset."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.data

    def split_data(self, target_column="pts", test_size=0.2, random_state=42):
        """Split data for predicting player points."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        train_set, test_set = train_test_split(
            self.data, test_size=test_size, random_state=random_state
        )
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame):
        """Save datasets to Unity Catalog."""
        # Convert pandas to spark dataframes with timestamp
        train_spark = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        test_spark = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        # Save to Unity Catalog
        train_spark.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.nba_train_set"
        )
        test_spark.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.nba_test_set"
        )

    def enable_change_data_feed(self):
        for table in ["nba_train_set", "nba_test_set"]:
            table_path = f"{self.config.catalog_name}.{self.config.schema_name}.{table}"
            self.spark.sql(
                f"ALTER TABLE {table_path} "
                "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
            )
