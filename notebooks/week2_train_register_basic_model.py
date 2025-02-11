# Databricks notebook source
import mlflow
from pyspark.sql import SparkSession

from nba_analysis.config import ProjectConfig
from nba_analysis.models.basic_model import BasicModel

# COMMAND ----------
# Setup MLflow
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Initialize configs and spark
config = ProjectConfig.from_yaml("../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags = {"git_sha": "your-git-sha", "branch": "week2"}

# COMMAND ----------
# Initialize and train model
basic_model = BasicModel(config=config, tags=tags, spark=spark)
basic_model.load_data()
basic_model.prepare_features()

# COMMAND ----------
# Train and log model
basic_model.train()
basic_model.log_model()

# COMMAND ----------
# Load the model from the latest run
run_id = mlflow.search_runs(
    experiment_names=[config.experiment_name_basic], filter_string="tags.branch='week2'"
).run_id[0]
model = mlflow.sklearn.load_model(f"runs:/{run_id}/nba-points-prediction-model")

# COMMAND ----------
# Retrieve dataset for the current run
basic_model.retrieve_current_run_dataset()

# COMMAND ----------
# Retrieve metadata for the current run
basic_model.retrieve_current_run_metadata()

# COMMAND ----------
# Register model
basic_model.register_model()

# COMMAND ----------
# Make predictions
test_set = spark.table(
    f"{config.catalog_name}.{config.schema_name}.nba_test_set"
).limit(10)
X_test = test_set.drop(config.target_column).toPandas()
predictions = basic_model.model.predict(X_test[basic_model.features])
print("Predictions:", predictions)
