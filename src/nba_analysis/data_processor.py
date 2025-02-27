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
        file_path = self.config.input_data
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

            train_spark.write.mode("overwrite").saveAsTable(train_table)
            test_spark.write.mode("overwrite").saveAsTable(test_table)

            # Enable change data feed
            for table in [train_table, test_table]:
                spark.sql(f"""
                ALTER TABLE {table}
                SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
                """)

        return self

    def make_synthetic_data(self, num_rows=10):
        """Generate synthetic NBA player data"""
        synthetic_data = pd.DataFrame()

        for column in self.data.columns:
            if column == self.config.id_column:
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

            elif pd.api.types.is_numeric_dtype(self.data[column]):
                # For numeric columns (stats, measurements)
                synthetic_data[column] = np.random.uniform(
                    self.data[column].min(), self.data[column].max(), num_rows
                )
            elif pd.api.types.is_string_dtype(self.data[column]):
                # For categorical columns (team, college, etc.)
                synthetic_data[column] = np.random.choice(
                    self.data[column].unique(),
                    num_rows,
                    p=self.data[column].value_counts(normalize=True),
                )
            else:
                synthetic_data[column] = np.random.choice(self.data[column], num_rows)

        return synthetic_data
