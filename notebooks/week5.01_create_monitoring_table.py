# Databricks notebook source
# MAGIC %md
# MAGIC ## NBA Points Prediction - Model Monitoring Setup

# COMMAND ----------
import datetime
import itertools
import time

import pandas as pd
import requests
from databricks.sdk import WorkspaceClient
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from src.nba_analysis.config import Config
from src.nba_analysis.data_processor import generate_synthetic_data
from src.nba_analysis.monitoring import create_or_refresh_monitoring

spark = SparkSession.builder.getOrCreate()
# Initialize dbutils
dbutils = DBUtils(spark)

token = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)
host = spark.conf.get("spark.databricks.workspaceUrl")


# Load configuration
config = Config.from_yaml(config_path="project_config.yml", env="dev")

# Load train and test data
train_set = spark.table(
    f"{config.catalog_name}.{config.schema_name}.train_set"
).toPandas()
test_set = spark.table(
    f"{config.catalog_name}.{config.schema_name}.test_set"
).toPandas()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Feature Importance Analysis

# COMMAND ----------


# Encode categorical variables
def preprocess_data(df):
    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders


train_encoded, label_encoders = preprocess_data(train_set)

# Define features and target
features = train_encoded.drop(columns=[config.target])
target = train_encoded[config.target]

# Train a Random Forest model for feature importance
model = RandomForestRegressor(random_state=42)
model.fit(features, target)

# Identify the most important features
feature_importances = pd.DataFrame(
    {"Feature": features.columns, "Importance": model.feature_importances_}
).sort_values(by="Importance", ascending=False)

print("Top 5 important features:")
print(feature_importances.head(5))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Generate Synthetic Data with Drift

# COMMAND ----------

# Generate data with drift
inference_data_skewed = generate_synthetic_data(train_set, drift=True, num_rows=200)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Save Synthetic Data for Testing

# COMMAND ----------
# Save the synthetic data
inference_data_skewed_spark = spark.createDataFrame(inference_data_skewed).withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

inference_data_skewed_spark.write.mode("overwrite").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.inference_data_skewed"
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Send Test Traffic to Model Endpoint

# COMMAND ----------
workspace = WorkspaceClient()
token = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)
host = spark.conf.get("spark.databricks.workspaceUrl")

# Required columns for inference
required_columns = config.num_features + config.cat_features

# Sample records
test_set_records = test_set[required_columns + ["player_name"]].to_dict(
    orient="records"
)
skewed_records = inference_data_skewed[required_columns + ["player_name"]].to_dict(
    orient="records"
)


# COMMAND ----------
# Function to send requests to endpoint
def send_request_https(dataframe_record):
    model_serving_endpoint = (
        f"https://{host}/serving-endpoints/nba-points-model-serving/invocations"
    )
    response = requests.post(
        model_serving_endpoint,
        headers={"Authorization": f"Bearer {token}"},
        json={"dataframe_records": [dataframe_record]},
    )
    return response


# COMMAND ----------
# Send normal test data
print("Sending normal test data...")
end_time = datetime.datetime.now() + datetime.timedelta(
    minutes=5
)  # 5 minutes of traffic
for index, record in enumerate(itertools.cycle(test_set_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Test data request {index}")
    try:
        response = send_request_https(record)
        print(f"Response status: {response.status_code}")
    except Exception as e:
        print(f"Error: {str(e)}")
    time.sleep(0.2)  # Slight delay between requests

# COMMAND ----------
# Send skewed data to simulate drift
print("Sending skewed data...")
end_time = datetime.datetime.now() + datetime.timedelta(
    minutes=5
)  # 5 minutes of traffic
for index, record in enumerate(itertools.cycle(skewed_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Skewed data request {index}")
    try:
        response = send_request_https(record)
        print(f"Response status: {response.status_code}")
    except Exception as e:
        print(f"Error: {str(e)}")
    time.sleep(0.2)  # Slight delay between requests

# COMMAND ----------
# MAGIC %md
# MAGIC ## Create and Refresh Monitoring Tables

# COMMAND ----------


workspace = WorkspaceClient()

# Load configuration (use production config for monitoring)
config = Config.from_yaml(config_path="project_config.yml", env="prd")

# Create or refresh monitoring tables
monitoring_table = create_or_refresh_monitoring(
    config=config, spark=spark, workspace=workspace
)

print(f"Monitoring table created: {monitoring_table}")
