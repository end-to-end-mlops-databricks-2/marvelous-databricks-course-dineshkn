#!/usr/bin/env python
"""
Script for preprocessing NBA player data
"""

import argparse
import logging

import yaml
from pyspark.sql import SparkSession

from nba_analysis.config import Config
from nba_analysis.data_processor import DataProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True, help="Root path for the project")
    parser.add_argument("--env", default="dev", help="Environment (dev/acc/prd)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Setup config and Spark
    config_path = f"{args.root_path}/files/project_config.yml"
    config = Config.from_yaml(config_path=config_path, env=args.env)
    logger.info("Configuration loaded:")
    logger.info(yaml.dump(config.model_dump(), default_flow_style=False))

    spark = SparkSession.builder.getOrCreate()

    # Create DataProcessor with None input_df to load from configuration
    logger.info("Initializing DataProcessor")
    data_processor = DataProcessor(input_df=None, config=config)

    # If your DataProcessor has a preprocess method, call it
    if hasattr(data_processor, "preprocess"):
        logger.info("Preprocessing data")
        data_processor.preprocess()

    # If your DataProcessor has a split_data method, call it
    if hasattr(data_processor, "split_data"):
        logger.info("Splitting data into train/test sets")
        train_df, test_df = data_processor.split_data(test_size=0.2, random_state=42)
        logger.info(
            f"Train shape: {train_df.shape if train_df is not None else 'unknown'}"
        )
        logger.info(
            f"Test shape: {test_df.shape if test_df is not None else 'unknown'}"
        )

    # If your DataProcessor has a save method, call it
    if hasattr(data_processor, "save_to_catalog"):
        logger.info("Saving data to catalog")
        try:
            train_table_name = f"{config.catalog_name}.{config.schema_name}.train_set"
            test_table_name = f"{config.catalog_name}.{config.schema_name}.test_set"

            # Check if the DataFrame is already a SparkDataFrame
            if hasattr(train_df, "write"):
                train_df.write.mode("overwrite").saveAsTable(train_table_name)
            else:
                spark.createDataFrame(train_df).write.mode("overwrite").saveAsTable(
                    train_table_name
                )

            if hasattr(test_df, "write"):
                test_df.write.mode("overwrite").saveAsTable(test_table_name)
            else:
                spark.createDataFrame(test_df).write.mode("overwrite").saveAsTable(
                    test_table_name
                )

            logger.info("Data saved to catalog successfully")
        except Exception as e:
            logger.error(f"Error saving data to catalog: {str(e)}")

    logger.info("✅ Data preprocessing complete")


if __name__ == "__main__":
    main()
