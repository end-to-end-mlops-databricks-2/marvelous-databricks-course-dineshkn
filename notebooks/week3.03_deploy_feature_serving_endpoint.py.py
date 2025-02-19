# Databricks notebook source
import os
import time
import mlflow
import pandas as pd
import requests
from databricks import feature_engineering
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from src.nba_analysis.config import ProjectConfig
from src.nba_analysis.serving.feature_serving import FeatureServing

# COMMAND ----------

# Load project config
config = ProjectConfig.from_yaml("../project_config.yml")

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
fe = feature_engineering.FeatureEngineeringClient()
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils() \
    .notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

catalog_name = config.catalog_name
schema_name = config.schema_name
feature_table_name = f"{catalog_name}.{schema_name}.nba_points_preds"
feature_spec_name = f"{catalog_name}.{schema_name}.return_predictions"
endpoint_name = "nba-points-feature-serving"

# COMMAND ----------

# Load data and make predictions
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
df = pd.concat([train_set, test_set])

# COMMAND ----------

# Load the model with pipeline
model = mlflow.sklearn.load_model(
    f"models:/{catalog_name}.{schema_name}.nba_points_model_basic@latest-model"
)

# COMMAND ----------

# Aggregate predictions by player
preds_df = df[["player_name", "team_abbreviation", "age"]]
preds_df["Predicted_Points"] = model.predict(df[config.num_features])
preds_df = preds_df.groupby("player_name").agg({
    "Predicted_Points": "mean",
    "team_abbreviation": "last",
    "age": "last"
}).reset_index()
preds_df = spark.createDataFrame(preds_df)

# COMMAND ----------

# Create feature table
fe.create_table(
    name=feature_table_name,
    primary_keys=["player_name"],
    df=preds_df,
    description="NBA Points predictions feature table"
)

# COMMAND ----------

# Enable change data feed
spark.sql(f"""
    ALTER TABLE {feature_table_name}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

# Initialize feature store manager
feature_serving = FeatureServing(
    feature_table_name=feature_table_name,
    feature_spec_name=feature_spec_name,
    endpoint_name=endpoint_name
)

# COMMAND ----------

# Create online table
feature_serving.create_online_table()

# COMMAND ----------

# Create feature spec
feature_serving.create_feature_spec()

# COMMAND ----------

# Deploy feature serving endpoint
feature_serving.deploy_or_update_serving_endpoint()

# COMMAND ----------

# After feature_serving.deploy_or_update_serving_endpoint()
print("Available endpoints:", [
    endpoint.name
    for endpoint in feature_serving.workspace.serving_endpoints.list()
])

# Check endpoint status
endpoint = feature_serving.workspace.serving_endpoints.get(
    feature_serving.endpoint_name
)
print(f"Endpoint state: {endpoint.state}")

# Wait for endpoint to be ready
print("Waiting for endpoint to be ready...")
while endpoint.state.ready == "NOT_READY":
    time.sleep(10)  # Wait 10 seconds between checks
    endpoint = feature_serving.workspace.endpoints.get(feature_serving.endpoint_name)
    print(f"Current state: {endpoint.state}")
    
print("Endpoint ready, attempting request...")

# COMMAND ----------

# Test the endpoint
start_time = time.time()
serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"
response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
    json={"dataframe_records": [{"player_name": "Stephen Curry"}]},
)
end_time = time.time()
execution_time = end_time - start_time

# COMMAND ----------

print("Response status:", response.status_code)
print("Response text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

# another way to call the endpoint
response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
    json={"dataframe_split": {"columns": ["player_name"], "data": [["Stephen Curry"]]}}
)

# COMMAND ----------


