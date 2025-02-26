#!/usr/bin/env python
"""
Script for preprocessing NBA player data
"""

import argparse
import logging

import pandas as pd
import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from nba_analysis.config import Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True, help="Root path for the project")
    parser.add_argument("--env", default="dev", help="Environment (dev/acc/prd)")
    return parser.parse_args()


def main():
    logger = logging.getLogger(__name__)
    args = parse_args()
    root_path = args.root_path
    config_path = f"{root_path}/files/project_config.yml"
    config = Config.from_yaml(config_path=config_path, env=args.env)

    logger.info("Configuration loaded:")
    logger.info(yaml.dump(config, default_flow_style=False))
    spark = SparkSession.builder.getOrCreate()

    print(f"Loading data from: {config.data}")

    # Load and process data
    df = pd.read_csv(config.data)

    # Clean and preprocess
    print("Preprocessing data...")

    # Convert numerical features
    for col in config.num_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle missing values
    df = df.fillna(
        {
            "college": "Unknown",
            "country": "Unknown",
            "draft_year": "0",
            "draft_round": "0",
            "draft_number": "0",
        }
    )

    # Split into train/test
    print("Splitting data into train/test sets...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save to catalog
    catalog_name = config.catalog_name
    schema_name = config.schema_name

    print(f"Saving train/test data to {catalog_name}.{schema_name}")

    # Save train set
    train_spark = spark.createDataFrame(train_df).withColumn(
        "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
    )
    train_spark.write.mode("overwrite").saveAsTable(
        f"{catalog_name}.{schema_name}.train_set"
    )

    # Save test set
    test_spark = spark.createDataFrame(test_df).withColumn(
        "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
    )
    test_spark.write.mode("overwrite").saveAsTable(
        f"{catalog_name}.{schema_name}.test_set"
    )

    # Enable change data feed
    for table in ["train_set", "test_set"]:
        spark.sql(f"""
        ALTER TABLE {catalog_name}.{schema_name}.{table}
        SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
        """)

    print("✅ Data preprocessing complete and tables created")


if __name__ == "__main__":
    main()
