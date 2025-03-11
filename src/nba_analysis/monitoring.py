"""
Model monitoring utilities for NBA points prediction.
"""

from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from loguru import logger
from pyspark.sql import functions as F
from pyspark.sql.functions import concat_ws
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
        table_name = f"{config.catalog_name}.{config.schema_name}.nba-points-model-serving_payload"
        inf_table = spark.sql(f"SELECT * FROM `{table_name}`")
        logger.info(f"Found model serving payload table: {table_name}")
    except Exception as e:
        logger.warning(
            f"No inference logs found. Creating empty monitoring table. Error: {str(e)}"
        )
        return create_empty_monitoring_table(config, spark, workspace)

    # Define request schema for parsing data
    request_schema = StructType(
        [
            StructField(
                "dataframe_records",
                ArrayType(
                    StructType(
                        [
                            StructField("team_abbreviation", StringType(), True),
                            StructField("age", DoubleType(), True),
                            StructField("player_height", DoubleType(), True),
                            StructField("player_weight", DoubleType(), True),
                            StructField("gp", DoubleType(), True),
                        ]
                    )
                ),
                True,
            )
        ]
    )

    # Define response schema for parsing model predictions
    response_schema = StructType(
        [
            StructField("predictions", ArrayType(DoubleType()), True),
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

    # Parse request and response JSON
    try:
        inf_table_parsed = inf_table.withColumn(
            "parsed_request", F.from_json(F.col("request"), request_schema)
        ).withColumn("parsed_response", F.from_json(F.col("response"), response_schema))
    except Exception as e:
        logger.error(f"Error parsing request/response JSON: {str(e)}")
        return create_empty_monitoring_table(config, spark, workspace)

    # Extract a unique player identifier (player_id)
    try:
        inf_table_with_id = inf_table_parsed.withColumn(
            "player_id",
            concat_ws(
                "_",
                F.get_json_object(
                    F.col("request"), "$.dataframe_records[0].team_abbreviation"
                ),
                F.round(
                    F.get_json_object(F.col("request"), "$.dataframe_records[0].age"), 1
                ),
                F.get_json_object(
                    F.col("request"), "$.dataframe_records[0].player_height"
                ),
                F.get_json_object(
                    F.col("request"), "$.dataframe_records[0].player_weight"
                ),
                F.get_json_object(F.col("request"), "$.dataframe_records[0].gp"),
            ),
        )
    except Exception as e:
        logger.error(f"Error extracting player identifier: {str(e)}")
        return create_empty_monitoring_table(config, spark, workspace)

    # Select relevant fields for monitoring
    df_final = inf_table_with_id.select(
        F.from_unixtime(F.col("timestamp_ms") / 1000)
        .cast("timestamp")
        .alias("timestamp"),
        "databricks_request_id",
        "execution_time_ms",
        "player_id",
        F.col("parsed_response.predictions")[0].alias("prediction"),
        F.lit("nba_points_model_basic").alias("model_name"),
    )

    # Check if test_set exists
    try:
        test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")

        df_final_with_truth = df_final.join(
            test_set.select(
                concat_ws(
                    "_",
                    F.col("team_abbreviation"),
                    F.round(F.col("age"), 1),
                    F.col("player_height"),
                    F.col("player_weight"),
                    F.col("gp"),
                ).alias("player_id"),
                "pts",
            ),
            on="player_id",
            how="left",
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

    # Write to monitoring table
    monitoring_table = (
        f"{config.catalog_name}.{config.schema_name}.nba_model_monitoring"
    )

    try:
        df_with_metrics.write.format("delta").mode("append").saveAsTable(
            monitoring_table
        )
        logger.info(
            f"Successfully wrote {df_with_metrics.count()} "
            f"records to {monitoring_table}"
        )
    except Exception as e:
        logger.warning(
            f"Error writing to table, attempting to create it first: {str(e)}"
        )
        df_with_metrics.write.format("delta").mode("overwrite").saveAsTable(
            monitoring_table
        )

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

    schema = StructType(
        [
            StructField("timestamp", StringType(), True),
            StructField("databricks_request_id", StringType(), True),
            StructField("execution_time_ms", IntegerType(), True),
            StructField("player_id", StringType(), True),
            StructField("prediction", DoubleType(), True),
            StructField("model_name", StringType(), True),
            StructField("actual_points", DoubleType(), True),
            StructField("error", DoubleType(), True),
            StructField("abs_error", DoubleType(), True),
            StructField("squared_error", DoubleType(), True),
        ]
    )

    empty_df = spark.createDataFrame([], schema)

    empty_df.write.format("delta").mode("overwrite").saveAsTable(monitoring_table)
    logger.info(f"Created empty monitoring table: {monitoring_table}")

    return monitoring_table
