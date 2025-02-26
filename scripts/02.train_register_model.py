#!/usr/bin/env python
"""
Script for training and registering the NBA points prediction model
"""

import argparse

import mlflow
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from nba_analysis.config import Config, Tags
from nba_analysis.models.basic_model import BasicModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path",
        action="store",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--env",
        action="store",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--git_sha",
        action="store",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--branch",
        action="store",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--job_run_id", action="store", default=None, type=str, required=False
    )
    return parser.parse_args()


def main():
    # Configure MLflow
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    # Parse arguments
    args = parse_args()
    config_path = f"{args.root_path}/project_config.yml"

    # Load configuration
    config = Config.from_yaml(config_path=config_path, env=args.env)
    spark = SparkSession.builder.getOrCreate()
    dbutils = DBUtils(spark)

    # Set up tags
    tags_dict = {
        "git_sha": args.git_sha,
        "branch": args.branch,
        "job_run_id": args.job_run_id if args.job_run_id else "local",
    }
    tags = Tags(**tags_dict)

    # Initialize model
    logger.info("Initializing model")
    basic_model = BasicModel(config=config, tags=tags, spark=spark)

    # Set experiment
    mlflow.set_experiment(config.experiment_name)

    # Load and prepare data
    logger.info("Loading data")
    basic_model.load_data()
    basic_model.prepare_features()

    # Train model
    logger.info("Training model")
    basic_model.train()
    basic_model.log_model()

    # Evaluate model
    logger.info("Evaluating model")
    test_set = spark.table(
        f"{config.catalog_name}.{config.schema_name}.test_set"
    ).limit(100)

    # Check if model has improved over previous versions
    model_improved = True  # In future, implement proper model evaluation

    if model_improved:
        # Register the model
        latest_version = basic_model.register_model()
        logger.info(f"New model registered with version: {latest_version}")

        # Set task values for downstream tasks
        dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
        dbutils.jobs.taskValues.set(key="model_updated", value=1)
    else:
        dbutils.jobs.taskValues.set(key="model_updated", value=0)
        logger.info("Model did not improve, skipping registration")

    logger.info("✅ Training and evaluation complete")


if __name__ == "__main__":
    main()
