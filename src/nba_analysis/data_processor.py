import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split


class DataProcessor:
    """
    Data Processor for NBA analysis.
    Pre-processes data for ML model training and evaluation.
    """

    def __init__(self, input_df=None, config=None):
        """Initialize with either a DataFrame or config to load data"""
        self.config = config

        # Store data
        if input_df is not None:
            self.data = input_df
        else:
            self.load_data()

        # Initialize train/test attributes
        self.train_df = None
        self.test_df = None

    def load_data(self):
        """Load data from the file specified in config"""
        file_path = self.config.data
        self.data = pd.read_csv(file_path)
        return self

    def preprocess(self):
        """Preprocess the loaded data"""
        # Handle numeric features
        for col in self.config.num_features:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce")

        # Handle categorical features
        for col in self.config.cat_features:
            if col in self.data.columns:
                # You might add encoding here if needed
                pass

        # Handle missing values
        self.data = self.data.fillna(
            {
                "college": "Unknown",
                "country": "Unknown",
                "draft_year": "0",
                "draft_round": "0",
                "draft_number": "0",
            }
        )

        return self

    def split_data(self, test_size=0.2, random_state=42):
        """Split the data into training and test sets"""
        self.train_df, self.test_df = train_test_split(
            self.data, test_size=test_size, random_state=random_state
        )
        return self.train_df, self.test_df

    def save_to_catalog(self, spark=None):
        """Save the train and test sets to the catalog"""
        if spark is None:
            spark = SparkSession.builder.getOrCreate()

        if self.train_df is not None and self.test_df is not None:
            # Drop "Unnamed: 0" if it exists
            self.train_df = self.train_df.drop(columns=["Unnamed: 0"], errors="ignore")
            self.test_df = self.test_df.drop(columns=["Unnamed: 0"], errors="ignore")

            # Rename columns to remove invalid characters
            def clean_column_names(df):
                df.columns = [
                    col.replace(" ", "_")
                    .replace(";", "_")
                    .replace("{", "_")
                    .replace("}", "_")
                    .replace("(", "_")
                    .replace(")", "_")
                    .replace("\n", "_")
                    .replace("\t", "_")
                    .replace("=", "_")
                    for col in df.columns
                ]
                return df

            self.train_df = clean_column_names(self.train_df)
            self.test_df = clean_column_names(self.test_df)
            # Convert to Spark DataFrames with timestamp
            train_spark = spark.createDataFrame(self.train_df).withColumn(
                "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
            )

            test_spark = spark.createDataFrame(self.test_df).withColumn(
                "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
            )

            # Save to catalog
            train_table = (
                f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
            )
            test_table = (
                f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
            )

            # Drop existing tables to avoid schema conflicts
            spark.sql(f"DROP TABLE IF EXISTS {train_table}")
            spark.sql(f"DROP TABLE IF EXISTS {test_table}")

            train_spark.write.mode("overwrite").saveAsTable(train_table)
            test_spark.write.mode("overwrite").saveAsTable(test_table)

            # Enable change data feed
            for table in [train_table, test_table]:
                spark.sql(f"""
                ALTER TABLE {table}
                SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
                """)

        return self


def generate_synthetic_data(data, drift=False, num_rows=100):
    """
    Generate synthetic NBA player data, optionally with drift.

    Args:
    drift: Whether to introduce data drift
    num_rows: Number of rows to generate

    Returns:
    Pandas DataFrame with synthetic data
    """
    synthetic_data = pd.DataFrame()

    # For each column in the original dataframe
    for column in data.columns:
        if column == "player_name":
            # Generate random player names
            first_names = [
                "James",
                "Michael",
                "Kevin",
                "Stephen",
                "LeBron",
                "Kobe",
                "Shaq",
                "Magic",
            ]
            last_names = [
                "Johnson",
                "Jordan",
                "Bryant",
                "James",
                "Curry",
                "Durant",
                "Davis",
                "Thompson",
            ]

            synthetic_data[column] = [
                f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
                for _ in range(num_rows)
            ]

        elif pd.api.types.is_numeric_dtype(data[column]):
            # For numeric columns
            if column in ["age", "gp"]:  # Handle specific numeric columns
                synthetic_data[column] = np.random.randint(
                    data[column].min(), data[column].max() + 1, num_rows
                )
            else:
                synthetic_data[column] = np.random.normal(
                    data[column].mean(), data[column].std(), num_rows
                )

                # Ensure points are non-negative
                if column == "pts":
                    synthetic_data[column] = np.maximum(0, synthetic_data[column])

        elif pd.api.types.is_categorical_dtype(
            data[column]
        ) or pd.api.types.is_object_dtype(data[column]):
            # For categorical columns
            synthetic_data[column] = np.random.choice(
                data[column].unique(),
                num_rows,
                p=data[column].value_counts(normalize=True),
            )
        else:
            # For other types
            synthetic_data[column] = np.random.choice(data[column], num_rows)

    # Convert relevant columns to integers
    int_columns = {"gp", "age"}
    for col in int_columns.intersection(data.columns):
        synthetic_data[col] = synthetic_data[col].astype(np.int32)

    # Apply drift if requested
    if drift:
        # Skew important features
        top_features = ["player_height", "player_weight", "ast"]
        for feature in top_features:
            if feature in synthetic_data.columns:
                synthetic_data[feature] = synthetic_data[feature] * 1.5

        # Generate different distribution for points
        if "pts" in synthetic_data.columns:
            # Increase average points by 50%
            synthetic_data["pts"] = synthetic_data["pts"] * 1.5

        # Create unusual team distributions
        if "team_abbreviation" in synthetic_data.columns:
            teams = synthetic_data["team_abbreviation"].unique()
            if len(teams) > 2:
                # Concentrate 80% of players on just 2 teams
                team_probs = [0.4, 0.4] + [0.2 / (len(teams) - 2)] * (len(teams) - 2)
                synthetic_data["team_abbreviation"] = np.random.choice(
                    teams, num_rows, p=team_probs
                )

    return synthetic_data
