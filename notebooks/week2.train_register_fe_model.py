# Databricks notebook source
import sys
from pathlib import Path

import mlflow
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split

from src.nba_analysis.config import ProjectConfig
from src.nba_analysis.models.feature_lookup_model import FeatureLookUpModel

# COMMAND ----------
repo_root = Path().resolve().parent
sys.path.append(str(repo_root))

# COMMAND ----------
# Setup MLflow
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
# COMMAND ----------
mlflow.set_experiment("/Shared/nba-points-fe")

# COMMAND ----------
# Initialize configs and spark
config = ProjectConfig.from_yaml("../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags = {"git_sha": "your-git-sha", "branch": "week2"}

# COMMAND ----------
# Load raw data
data = spark.read.csv(config.input_data, header=True, inferSchema=True)

# Split the data
train_data, test_data = train_test_split(
    data.toPandas(), test_size=0.2, random_state=42
)

# Convert to Spark DataFrames and save as tables
spark.createDataFrame(train_data).write.mode("overwrite").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.train_set"
)
spark.createDataFrame(test_data).write.mode("overwrite").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.test_set"
)

# COMMAND ----------
# Initialize model with feature engineering
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# COMMAND ----------
# Create feature table
fe_model.create_feature_table()

# COMMAND ----------
# Define player performance feature function
fe_model.define_feature_function()

# COMMAND ----------
# Load data
fe_model.load_data()

# COMMAND ----------
# Perform feature engineering
fe_model.feature_engineering()

# COMMAND ----------
# Train the model
fe_model.train()

# COMMAND ----------
# Register the model
fe_model.register_model()
