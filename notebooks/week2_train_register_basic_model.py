# Databricks notebook source
import sys
from pathlib import Path
from mlflow.models import infer_signature
repo_root = Path().resolve().parent
sys.path.append(str(repo_root))

# COMMAND ----------

import mlflow
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
from src.nba_analysis.config import ProjectConfig
from src.nba_analysis.models.basic_model import BasicModel

# COMMAND ----------

# Setup MLflow
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Set the experiment explicitly
mlflow.set_experiment("/Shared/nba-points-basic")

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

# Split the data
train_data, test_data = train_test_split(basic_model.data, test_size=0.2, random_state=42)

# Convert to Spark DataFrames and save as tables
spark.createDataFrame(train_data).write.mode("overwrite").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.nba_train_set"
)
spark.createDataFrame(test_data).write.mode("overwrite").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.nba_test_set"
)

# COMMAND ----------

# Train and log model
basic_model.train()

# COMMAND ----------

basic_model.log_model()

# COMMAND ----------

runs = mlflow.search_runs(
    experiment_names=["/Shared/nba-points-basic"], 
    filter_string="tags.branch='week2'"
)

if not runs.empty:
    run_id = runs.run_id[0]
else:
    run_id = None

# COMMAND ----------

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

# COMMAND ----------
# Make predictions
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.nba_test_set").limit(10)
X_test = test_set.drop(config.target).toPandas()
predictions = basic_model.model.predict(X_test[config.num_features])  # Use num_features from config
print("Predictions:", predictions)
