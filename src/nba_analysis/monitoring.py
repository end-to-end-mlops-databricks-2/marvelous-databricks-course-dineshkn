"""
Model monitoring utilities for NBA points prediction.
"""

from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from loguru import logger
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


def create_or_refresh_monitoring(config, spark, workspace):
    """
    Create or refresh model monitoring tables for NBA points prediction.
    Processes inference logs and joins with actual data to create a monitoring table.

    Args:
        config: Project configuration
        spark: Spark session
        workspace: Databricks workspace client

    Returns:
        str: Name of the monitoring table
    """
    logger.info("Creating or refreshing NBA points prediction monitoring...")

    # Check if inference logs table exists
    try:
        # First try with feature serving naming convention
        table_name = (
            f"{config.catalog_name}.{config.schema_name}."
            + "`nba-points-feature-serving_payload_payload`"
        )
        inf_table = spark.sql(f"SELECT * FROM {table_name}")
        logger.info(f"Found feature serving payload table: {table_name}")
    except Exception:
        try:
            # Fall back to model serving naming convention
            table_name = (
                f"{config.catalog_name}.{config.schema_name}."
                + "nba-points-model-serving_payload_payload"
            )
            inf_table = spark.sql(f"SELECT * FROM `{table_name}`")
            logger.info(f"Found model serving payload table: {table_name}")
        except Exception as e2:
            logger.warning(
                "No inference logs found. Creating empty monitoring table. "
                f"Error: {str(e2)}"
            )
            return create_empty_monitoring_table(config, spark, workspace)

    # Define request schema for parsing player data in the request
    request_schema = StructType(
        [
            # Format 1: dataframe_records with player objects
            StructField(
                "dataframe_records",
                ArrayType(StructType([StructField("player_name", StringType(), True)])),
                True,
            ),
            # Format 2: dataframe_split
            StructField(
                "dataframe_split",
                StructType(
                    [
                        StructField("columns", ArrayType(StringType()), True),
                        StructField("data", ArrayType(ArrayType(StringType())), True),
                    ]
                ),
                True,
            ),
        ]
    )

    # Define response schema for parsing model predictions
    response_schema = StructType(
        [
            StructField("predictions", ArrayType(DoubleType()), True),
            StructField(
                "features",
                StructType(
                    [
                        StructField("player_name", StringType(), True),
                        StructField("team_abbreviation", StringType(), True),
                        StructField("age", DoubleType(), True),
                        StructField("Predicted_Points", DoubleType(), True),
                    ]
                ),
                True,
            ),
            StructField(
                "databricks_output",
                StructType(
                    [
                        StructField("trace", StringType(), True),
                        StructField("databricks_request_id", StringType(), True),
                    ]
                ),
                True,
            ),
        ]
    )

    # Parse request and response JSON, with error handling
    try:
        inf_table_parsed = inf_table.withColumn(
            "parsed_request", F.from_json(F.col("request"), request_schema)
        ).withColumn("parsed_response", F.from_json(F.col("response"), response_schema))
    except Exception as e:
        logger.error(f"Error parsing request/response JSON: {str(e)}")
        return create_empty_monitoring_table(config, spark, workspace)

    # Extract player name from request - handling both formats
    try:
        inf_table_with_player = inf_table_parsed.withColumn(
            "player_name",
            F.when(
                F.col("parsed_request.dataframe_records").isNotNull(),
                F.col("parsed_request.dataframe_records")[0]["player_name"],
            )
            .when(
                F.col("parsed_request.dataframe_split.data").isNotNull(),
                F.when(
                    F.expr("size(parsed_request.dataframe_split.data) > 0"),
                    F.col("parsed_request.dataframe_split.data")[0][0],
                ).otherwise(None),
            )
            .otherwise(None),
        )
    except Exception as e:
        logger.error(f"Error extracting player name: {str(e)}")
        return create_empty_monitoring_table(config, spark, workspace)

    # Select the relevant fields for monitoring
    df_final = inf_table_with_player.select(
        F.from_unixtime(F.col("timestamp_ms") / 1000)
        .cast("timestamp")
        .alias("timestamp"),
        "databricks_request_id",
        "execution_time_ms",
        "player_name",
        F.col("parsed_response.predictions")[0].alias("prediction"),
        F.lit("nba_points_model_basic").alias("model_name"),
    )

    # Check if test_set exists
    try:
        # Join with actual values for ground truth comparison
        test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")

        df_final_with_truth = df_final.join(
            test_set.select("player_name", "pts"), on="player_name", how="left"
        ).withColumnRenamed("pts", "actual_points")

        # Calculate error metrics
        df_with_metrics = (
            df_final_with_truth.withColumn(
                "error", F.col("prediction") - F.col("actual_points")
            )
            .withColumn("abs_error", F.abs(F.col("error")))
            .withColumn("squared_error", F.pow(F.col("error"), 2))
            .withColumn("prediction", F.col("prediction").cast("double"))
            .withColumn("actual_points", F.col("actual_points").cast("double"))
        )
    except Exception as e:
        logger.warning(
            f"Could not join with test set: {str(e)}. Proceeding without ground truth."
        )
        df_with_metrics = (
            df_final.withColumn("actual_points", F.lit(None).cast("double"))
            .withColumn("error", F.lit(None).cast("double"))
            .withColumn("abs_error", F.lit(None).cast("double"))
            .withColumn("squared_error", F.lit(None).cast("double"))
        )

    # Check if train_set exists for player stats
    try:
        # Add additional features from NBA player data to enrich monitoring
        player_stats = spark.table(
            f"{config.catalog_name}.{config.schema_name}.train_set"
        )

        # Select a subset of relevant columns to join
        player_stats_columns = player_stats.columns
        columns_to_select = ["player_name"]

        # Only add columns that actually exist in the dataset
        for col_name in [
            "player_height",
            "player_weight",
            "gp",
            "reb",
            "ast",
            "net_rating",
        ]:
            if col_name in player_stats_columns:
                columns_to_select.append(col_name)

        player_stats_subset = player_stats.select(*columns_to_select)

        df_final_with_features = df_with_metrics.join(
            player_stats_subset, on="player_name", how="left"
        )
    except Exception as e:
        logger.warning(
            f"Could not join with player stats: {str(e)}. "
            "Proceeding without additional features."
        )
        df_final_with_features = df_with_metrics

    # Write to monitoring table - using the NBA model monitoring table name
    monitoring_table = (
        f"{config.catalog_name}.{config.schema_name}.nba_model_monitoring"
    )

    # Write to table with error handling
    try:
        df_final_with_features.write.format("delta").mode("append").saveAsTable(
            monitoring_table
        )
        logger.info(
            f"Successfully wrote {df_final_with_features.count()} "
            f"records to {monitoring_table}"
        )
    except Exception as e:
        # Table might not exist - try creating it first
        logger.warning(
            f"Error writing to table, attempting to create it first: {str(e)}"
        )
        df_final_with_features.write.format("delta").mode("overwrite").saveAsTable(
            monitoring_table
        )

    # Create or refresh the monitoring in Databricks Lakehouse Monitoring
    try:
        workspace.quality_monitors.get(monitoring_table)
        workspace.quality_monitors.run_refresh(table_name=monitoring_table)
        logger.info("Lakehouse monitoring table exists, refreshing.")
    except NotFound:
        create_monitoring_table(config=config, spark=spark, workspace=workspace)
        logger.info("Lakehouse monitoring table is created.")
    except Exception as e:
        logger.error(f"Error setting up monitoring: {str(e)}")

    logger.info("✅ NBA points model monitoring tables created/refreshed successfully")
    return monitoring_table


def create_monitoring_table(config, spark, workspace):
    """
    Create monitoring table for NBA points prediction.
    Configures the Databricks Lakehouse Monitoring for the NBA model.

    Args:
        config: Project configuration
        spark: Spark session
        workspace: Databricks workspace client
    """
    logger.info("Creating new NBA points monitoring table...")

    monitoring_table = (
        f"{config.catalog_name}.{config.schema_name}.nba_model_monitoring"
    )

    # Configure monitoring for the table with regression monitoring
    try:
        workspace.quality_monitors.create(
            table_name=monitoring_table,
            assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
            output_schema_name=f"{config.catalog_name}.{config.schema_name}",
            inference_log=MonitorInferenceLog(
                problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
                prediction_col="prediction",
                timestamp_col="timestamp",
                granularities=["30 minutes", "1 day"],
                model_id_col="model_name",
                label_col="actual_points",
            ),
        )

        # Enable change data feed for monitoring to track changes
        spark.sql(
            f"ALTER TABLE {monitoring_table} "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        logger.info(
            f"NBA model monitoring table {monitoring_table} configured successfully"
        )
    except Exception as e:
        logger.error(f"Error creating monitoring table: {str(e)}")
        raise


def create_empty_monitoring_table(config, spark, workspace):
    """
    Create an empty monitoring table when inference logs are not available.

    Args:
        config: Project configuration
        spark: Spark session
        workspace: Databricks workspace client

    Returns:
        str: Name of the monitoring table
    """
    logger.info("Creating empty NBA model monitoring table...")

    monitoring_table = (
        f"{config.catalog_name}.{config.schema_name}.nba_model_monitoring"
    )

    # Create empty dataframe with correct schema
    schema = StructType(
        [
            StructField(
                "timestamp",
                spark.sql("SELECT CURRENT_TIMESTAMP").schema[0].dataType,
                True,
            ),
            StructField("databricks_request_id", StringType(), True),
            StructField("execution_time_ms", IntegerType(), True),
            StructField("player_name", StringType(), True),
            StructField("prediction", DoubleType(), True),
            StructField("model_name", StringType(), True),
            StructField("actual_points", DoubleType(), True),
            StructField("error", DoubleType(), True),
            StructField("abs_error", DoubleType(), True),
            StructField("squared_error", DoubleType(), True),
            StructField("player_height", DoubleType(), True),
            StructField("player_weight", DoubleType(), True),
            StructField("gp", DoubleType(), True),
            StructField("reb", DoubleType(), True),
            StructField("ast", DoubleType(), True),
            StructField("net_rating", DoubleType(), True),
        ]
    )

    empty_df = spark.createDataFrame([], schema)

    # Create the table
    try:
        empty_df.write.format("delta").mode("overwrite").saveAsTable(monitoring_table)
        logger.info(f"Created empty monitoring table: {monitoring_table}")
    except Exception as e:
        logger.error(f"Error creating empty table: {str(e)}")

    try:
        # Try to set up monitoring anyway
        create_monitoring_table(config, spark, workspace)
    except Exception as e:
        logger.warning(f"Could not set up monitoring on empty table: {str(e)}")

    return monitoring_table
