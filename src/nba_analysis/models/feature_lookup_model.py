from datetime import datetime

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMRegressor
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class FeatureLookUpModel:
    def __init__(self, config, tags, spark: SparkSession):
        """Initialize the model with project configuration."""
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = (
            f"{self.catalog_name}.{self.schema_name}.player_features"
        )
        self.function_name = (
            f"{self.catalog_name}.{self.schema_name}.calculate_player_experience"
        )

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags

    def create_feature_table(self):
        """Create or replace the player_features table and populate it."""
        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (player_name STRING NOT NULL,
         avg_points DOUBLE,
         avg_rebounds DOUBLE, 
         avg_assists DOUBLE);
        """)
        self.spark.sql(
            f"ALTER TABLE {self.feature_table_name} "
            f"ADD CONSTRAINT player_pk PRIMARY KEY(player_name);"
        )
        self.spark.sql(
            f"ALTER TABLE {self.feature_table_name} "
            f"SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        # Populate feature table from training and test sets
        for dataset in ["train_set", "test_set"]:
            self.spark.sql(f"""
            INSERT INTO {self.feature_table_name} 
            SELECT player_name,
                   AVG(pts) as avg_points,
                   AVG(reb) as avg_rebounds,
                   AVG(ast) as avg_assists
            FROM {self.catalog_name}.{self.schema_name}.{dataset}
            GROUP BY player_name
            """)

        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self):
        """Define a function to calculate player's experience in years."""
        self.spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(draft_year STRING)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        from datetime import datetime
        try:
            return datetime.now().year - int(draft_year) if draft_year.isdigit() else 0
        except:
            return 0
        $$
        """)
        logger.info("✅ Feature function defined.")

    def load_data(self):
        """Load training and testing data."""
        self.train_set = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.train_set"
        )

        self.test_set = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.test_set"
        ).toPandas()

        logger.info("✅ Data successfully loaded.")

    def feature_engineering(self):
        """Perform feature engineering using feature lookups."""
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["avg_points", "avg_rebounds", "avg_assists"],
                    lookup_key="player_name",
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="player_experience",
                    input_bindings={"draft_year": "draft_year"},
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()

        # Calculate experience for test set
        self.test_set["player_experience"] = self.test_set["draft_year"].apply(
            lambda x: datetime.now().year - int(x) if str(x).isdigit() else 0
        )

        self.X_train = self.training_df[
            self.num_features + self.cat_features + ["player_experience"]
        ]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[
            self.num_features + self.cat_features + ["player_experience"]
        ]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train(self):
        """Train the model and log results to MLflow."""
        logger.info("🚀 Starting training...")

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)
            ],
            remainder="passthrough",
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LGBMRegressor(**self.parameters)),
            ]
        )

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            logger.info(f"📊 Mean Squared Error: {mse}")
            logger.info(f"📊 Mean Absolute Error: {mae}")
            logger.info(f"📊 R2 Score: {r2}")

            mlflow.log_param("model_type", "LightGBM with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)

            signature = infer_signature(self.X_train, y_pred)

            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                training_set=self.training_set,
                signature=signature,
            )

    def register_model(self):
        """Register model in Unity Catalog"""
        logger.info("🔄 Registering the model in UC...")

        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model-fe",
            name=f"{self.catalog_name}.{self.schema_name}.nba_points_model_fe",
            tags=self.tags,
        )

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.nba_points_model_fe",
            alias="latest-model",
            version=latest_version,
        )
        logger.info(f"✅ Model registered as version {latest_version}.")

    def load_latest_model_and_predict(self, X):
        """
        Load the trained model from MLflow using Feature Engineering Client
        and make predictions.
        """
        model_uri = (
            f"models:/{self.catalog_name}.{self.schema_name}.nba_points_model_fe"
            f"@latest-model"
        )
        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions
