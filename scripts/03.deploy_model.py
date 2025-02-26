#!/usr/bin/env python
"""
Script for deploying NBA model with feature lookups
"""

import argparse

from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from nba_analysis.config import Config
from nba_analysis.serving.feature_serving import FeatureServing
from nba_analysis.serving.model_serving import ModelServing


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
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup configuration and Spark
    config_path = f"{args.root_path}/files/project_config.yml"
    spark = SparkSession.builder.getOrCreate()
    dbutils = DBUtils(spark)

    # Try to get model version from previous task (if available)
    try:
        model_version = dbutils.jobs.taskValues.get(
            taskKey="train_register_model", key="model_version"
        )
        logger.info(f"Retrieved model version: {model_version}")
    except Exception:
        model_version = "latest-model"
        logger.info(f"Using default model version: {model_version}")

    # Load project config
    config = Config.from_yaml(config_path=config_path, env=args.env)
    logger.info("Loaded config file.")

    catalog_name = config.catalog_name
    schema_name = config.schema_name

    # Deploy model serving endpoint
    logger.info("Deploying model serving endpoint...")
    model_serving = ModelServing(
        model_name=f"{catalog_name}.{schema_name}.nba_points_model_basic",
        endpoint_name=f"nba-points-model-serving-{args.env}",
    )
    model_serving.deploy_or_update_serving_endpoint(version=model_version)
    logger.info("Model serving endpoint deployed/updated")

    # Set up feature table and serving
    logger.info("Setting up feature serving...")
    feature_table_name = f"{catalog_name}.{schema_name}.nba_points_preds"
    feature_spec_name = f"{catalog_name}.{schema_name}.return_predictions"
    feature_endpoint_name = f"nba-points-feature-serving-{args.env}"

    feature_serving = FeatureServing(
        feature_table_name=feature_table_name,
        feature_spec_name=feature_spec_name,
        endpoint_name=feature_endpoint_name,
    )

    # Update online table
    feature_serving.create_online_table()
    logger.info("Created/updated online feature table")

    # Deploy feature serving endpoint
    feature_serving.deploy_or_update_serving_endpoint()
    logger.info("Feature serving endpoint deployed/updated")

    logger.info("✅ All deployments complete")


if __name__ == "__main__":
    main()
