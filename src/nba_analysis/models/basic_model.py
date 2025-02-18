import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BasicModel:
    def __init__(self, config, tags, spark=None):
        self.config = config
        self.tags = tags
        self.spark = spark
        self.data = None
        self.X = None
        self.y = None
        self.model = None

    def load_data(self):
        """Load data from either Databricks or local file"""
        try:
            print("Attempting to load from Databricks volume...")
            # Add header=True and inferSchema=True
            self.data = self.spark.read.csv(
                self.config.input_data, header=True, inferSchema=True
            ).toPandas()
            print("Successfully loaded data from Databricks volume")
        except Exception as e:
            print(f"Error loading from Databricks: {str(e)}")
            raise  # Re-raise the exception instead of falling back to local file
        print(f"Data loaded with shape: {self.data.shape}")
        return self

    def prepare_features(self):
        """Prepare features for training"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.X = self.data[self.config.num_features]
        self.y = self.data[self.config.target]
        print(f"Prepared features: {len(self.config.num_features)} features")
        return self

    def train(self):
        """Train the model"""
        if self.X is None or self.y is None:
            raise ValueError("Features not prepared. Call prepare_features() first.")

        # Get parameters from config
        params = self.config.parameters

        # Create pipeline
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        random_state=42,
                    ),
                ),
            ]
        )

        # Train
        self.model.fit(self.X, self.y)
        print("Model training completed!")
        return self

    def predict(self, X):
        """Make predictions with trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def log_model(self):
        """Log the trained model to MLflow"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        with mlflow.start_run(tags=self.tags) as run:
            # Disable MLflow's autologging to prevent duplicate runs
            mlflow.autolog(disable=True)
            self.run_id = run.info.run_id  # Store the run ID
            # Log parameters
            mlflow.log_params(self.config.parameters)

            # Log metrics
            train_score = self.model.score(self.X, self.y)
            mlflow.log_metric("train_r2", train_score)

            # Infer model signature
            y_pred = self.model.predict(self.X)  # Get predictions on full dataset
            signature = mlflow.models.infer_signature(
                self.X, y_pred
            )  # Infer input-output schema

            # Log model with signature
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="nba-points-prediction-model",
                signature=signature,  # Include the required signature
                registered_model_name="nba_points_predictor",
            )
            print(f"✅ Model logged successfully in run: {self.run_id}")

    def register_model(self):
        """Register model in Unity Catalog"""
        logger.info("🔄 Registering the model in UC...")

        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/nba-points-prediction-model",
            name=f"{self.config.catalog_name}.{self.config.schema_name}.nba_points_model_basic",
            tags=self.tags,
        )

        logger.info(f"✅ Model registered as version {registered_model.version}.")

        # Set the latest version alias
        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.config.catalog_name}.{self.config.schema_name}.nba_points_model_basic",
            alias="latest-model",
            version=registered_model.version,
        )

    def retrieve_current_run_dataset(self):
        """Retrieve the dataset used in the current run"""
        if not hasattr(self, "run_id"):
            raise ValueError("No run ID found. Model must be logged first.")

        client = mlflow.tracking.MlflowClient()
        run = client.get_run(self.run_id)
        return run.data

    def retrieve_current_run_metadata(self):
        """Retrieve metadata for the current run"""
        if not hasattr(self, "run_id"):
            raise ValueError("No run ID found. Model must be logged first.")

        client = mlflow.tracking.MlflowClient()
        run = client.get_run(self.run_id)
        return run.info
