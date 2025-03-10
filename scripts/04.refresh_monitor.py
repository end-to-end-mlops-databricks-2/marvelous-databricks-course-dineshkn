#!/usr/bin/env python
"""
Script to refresh NBA points prediction model monitoring tables.
"""

import argparse
import logging
import os

from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

from nba_analysis.config import ProjectConfig
from nba_analysis.monitoring import create_or_refresh_monitoring

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function to refresh NBA model monitoring."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Refresh NBA model monitoring tables")
    parser.add_argument("--root_path", required=True, help="Root path of the project")
    parser.add_argument("--env", required=True, help="Environment (dev, stg, prd)")
    args = parser.parse_args()

    logger.info(f"Starting NBA model monitoring refresh in {args.env} environment")

    # Initialize Spark session and Databricks workspace client
    spark = SparkSession.builder.getOrCreate()
    workspace = WorkspaceClient()

    # Load configuration
    config_path = os.path.join(args.root_path, "project_config.yml")
    config = ProjectConfig.from_yaml(config_path, env=args.env)

    # Run the monitoring refresh
    try:
        monitoring_table = create_or_refresh_monitoring(config, spark, workspace)
        logger.info(f"Successfully refreshed monitoring table: {monitoring_table}")
    except Exception as e:
        logger.error(f"Error refreshing monitoring table: {str(e)}")
        raise

    logger.info("NBA model monitoring refresh completed")


if __name__ == "__main__":
    main()
