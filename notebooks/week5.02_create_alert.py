# Databricks notebook source
# MAGIC %md
# MAGIC # NBA Points Prediction - Model Monitoring Alerts

# COMMAND ----------
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql

from src.nba_analysis.config import ProjectConfig

# COMMAND ----------
# Load configuration
config = ProjectConfig.from_yaml(config_path="project_config.yml", env="prd")
catalog_name = config.catalog_name
schema_name = config.schema_name

# Initialize Workspace client
w = WorkspaceClient()
srcs = w.data_sources.list()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Create Alert for High Mean Absolute Error (MAE)

# COMMAND ----------
# Define query to check percentage of MAE readings above threshold
alert_query = f"""
SELECT 
  (COUNT(CASE WHEN abs_error > 30.0 THEN 1 END) * 100.0 / 
   COUNT(CASE WHEN abs_error IS NOT NULL 
              AND NOT isnan(abs_error) THEN 1 END)) AS percentage_high_mae
FROM {catalog_name}.{schema_name}.nba_model_monitoring
WHERE timestamp > CURRENT_TIMESTAMP() - INTERVAL 7 DAYS
"""

# Create the query
query_name = f"nba-points-alert-query-{time.time_ns()}"
query = w.queries.create(
    query=sql.CreateQueryRequestQuery(
        display_name=query_name,
        warehouse_id=srcs[0].warehouse_id,
        description="Alert on NBA points prediction model MAE",
        query_text=alert_query,
    )
)

# COMMAND ----------
# Create alert based on the query
alert_name = f"nba-points-mae-alert-{time.time_ns()}"
alert = w.alerts.create(
    alert=sql.CreateAlertRequestAlert(
        condition=sql.AlertCondition(
            operand=sql.AlertConditionOperand(
                type=sql.AlertConditionOperandType.NUMBER, double_value=40.0
            )  # Alert when >40% of readings exceed threshold
        ),
        display_name=alert_name,
        query_id=query.id,
    )
)

print(f"Created alert: {alert_name}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Create Alert for Data Drift

# COMMAND ----------
# Define query to check for data drift
drift_query = f"""
SELECT 
  (COUNT(CASE WHEN data_drift_score > 0.5 THEN 1 END) * 100.0 / 
   COUNT(CASE WHEN data_drift_score IS NOT NULL 
              AND NOT isnan(data_drift_score) THEN 1 END)) AS percentage_high_drift
FROM {catalog_name}.{schema_name}.nba_model_monitoring
WHERE timestamp > CURRENT_TIMESTAMP() - INTERVAL 7 DAYS
"""

# Create the query
drift_query_name = f"nba-points-drift-query-{time.time_ns()}"
drift_query_obj = w.queries.create(
    query=sql.CreateQueryRequestQuery(
        display_name=drift_query_name,
        warehouse_id=srcs[0].warehouse_id,
        description="Alert on NBA points prediction model data drift",
        query_text=drift_query,
    )
)

# Create alert based on the query
drift_alert_name = f"nba-points-drift-alert-{time.time_ns()}"
drift_alert = w.alerts.create(
    alert=sql.CreateAlertRequestAlert(
        condition=sql.AlertCondition(
            operand=sql.AlertConditionOperand(
                type=sql.AlertConditionOperandType.NUMBER, double_value=30.0
            )  # Alert when >30% of readings show high drift
        ),
        display_name=drift_alert_name,
        query_id=drift_query_obj.id,
    )
)

print(f"Created data drift alert: {drift_alert_name}")

# COMMAND ----------
w.queries.delete(id=query.id)
w.alerts.delete(id=alert.id)
