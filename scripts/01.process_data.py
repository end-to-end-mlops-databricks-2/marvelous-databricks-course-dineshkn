#!/usr/bin/env python
"""
Script for preprocessing NBA player data
"""

import argparse
import logging

import yaml

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

    # spark = SparkSession.builder.getOrCreate() # avoid pre-commit error

    # Create DataProcessor with None input_df to load from configuration
    logger.info("Initializing DataProcessor")
    data_processor = DataProcessor(input_df=None, config=config)

    # Generate new data from the original dataset
    synthetic_data = data_processor.make_synthetic_data(num_rows=100)
    logger.info("Synthetic data generated")

    # Initialise new DataProcessor
    new_processor = DataProcessor(input_df=synthetic_data, config=config)

    # Split data
    new_processor.split_data(test_size=0.3, random_state=42)
    logger.info("Train shape: %s", new_processor.train_df.shape)
    logger.info("Test shape: %s", new_processor.test_df.shape)

    # Save to catalog
    logger.info("Saving data to catalog")
    new_processor.save_to_catalog()


if __name__ == "__main__":
    main()
