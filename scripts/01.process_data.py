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
    root_path = args.root_path
    config_path = f"{root_path}/files/project_config.yml"

    # Setup config and Spark
    config = Config.from_yaml(
        config_path=config_path,
        env=args.env,
    )
    logging.info("Configuration loaded:")
    logging.info(yaml.dump(config, default_flow_style=False))

    spark = SparkSession.builder.getOrCreate()

    # Use the data processor to load the original dataset
    data_processor = DataProcessor(input_df=None, config=config)  # Pass spark here
    data_processor.preprocess()

    # Generate synthetic data
    synthetic_data = data_processor.make_synthetic_data(num_rows=100)
    logging.info("Synthetic data generated")

    # Later, also pass spark to the new processor
    new_processor = DataProcessor(input_df=synthetic_data, config=config)

    # Split data
    new_processor.split_data(test_size=0.2, random_state=42)
    logging.info(f"Train shape: {new_processor.train_df.shape}")
    logging.info(f"Test shape: {new_processor.test_df.shape}")

    # Save to catalog
    logging.info("Saving data to catalog")
    new_processor.save_traing_data_to_catalog()

    logging.info("✅ Data preprocessing complete and tables created")


if __name__ == "__main__":
    main()
