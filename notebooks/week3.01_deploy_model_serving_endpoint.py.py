# Databricks notebook source
import os
import time
from typing import Dict, List

import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from src.nba_analysis.config import ProjectConfig
from src.nba_analysis.serving.model_serving import ModelServing

# COMMAND ----------

# Initialize Spark and get environment variables
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
os.environ["DBR_TOKEN"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# Load project config
config = ProjectConfig.from_yaml("../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# Initialize model serving
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.nba_points_model_basic",
    endpoint_name="nba-points-model-serving",
)

# COMMAND ----------

# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()

# COMMAND ----------

# Create a sample request body with required columns
required_columns = [
    "age",
    "player_height",
    "player_weight",
    "gp",
    "reb",
    "ast",
    "net_rating",
    "oreb_pct",
    "dreb_pct",
    "usg_pct",
    "ts_pct",
    "ast_pct",
    "team_abbreviation",
    "college",
    "country",
    "draft_year",
    "draft_round",
    "draft_number",
]

# COMMAND ----------

# Sample records from test set
test_set = spark.table(
    f"{config.catalog_name}.{config.schema_name}.test_set"
).toPandas()

sampled_records = (
    test_set[required_columns].sample(n=100, replace=True).to_dict(orient="records")
)
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------


def call_endpoint(record: List[Dict]):
    """
    Calls the model serving endpoint with a given input record.
    """
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/nba-points-model-serving/invocations"
    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


# COMMAND ----------

# Test with one sample record
status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------

# Add this after model_serving.deploy_or_update_serving_endpoint()
print(
    "Available endpoints:",
    [endpoint.name for endpoint in model_serving.workspace.serving_endpoints.list()],
)

# COMMAND ----------

# Before calling the endpoint
print("Attempting to access endpoint: nba-points-model-serving")
print(
    f"Full endpoint URL: https://{os.environ['DBR_HOST']}/serving-endpoints/nba-points-model-serving/invocations"
)

# COMMAND ----------

# Check endpoint status
endpoint = model_serving.workspace.serving_endpoints.get(model_serving.endpoint_name)
print(f"Endpoint state: {endpoint.state}")

# Wait for it to be ready
while endpoint.state.ready == "NOT_READY":
    print("Waiting for endpoint to be ready...")
    time.sleep(10)  # Wait 10 seconds before checking again
    endpoint = model_serving.workspace.serving_endpoints.get(
        model_serving.endpoint_name
    )
    print(f"Current state: {endpoint.state}")

if endpoint.state.ready == "READY":
    print("Endpoint is ready!")
else:
    print(f"Endpoint is in state: {endpoint.state}")

# COMMAND ----------

# Check if model exists
print(f"Checking model: {model_serving.model_name}")
print(f"Latest version: {model_serving.get_latest_model_version()}")

# COMMAND ----------

# Load test
for i in range(len(dataframe_records)):
    call_endpoint(dataframe_records[i])
    time.sleep(0.2)
