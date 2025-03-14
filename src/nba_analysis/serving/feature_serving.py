import logging

from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput


class FeatureServing:
    def __init__(
        self, feature_table_name: str, feature_spec_name: str, endpoint_name: str
    ):
        """
        Initializes the Prediction Serving Manager.
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.feature_table_name = feature_table_name
        self.feature_table_name = feature_table_name
        self.workspace = WorkspaceClient()
        self.feature_spec_name = feature_spec_name
        self.endpoint_name = endpoint_name
        self.online_table_name = f"{self.feature_table_name}_online"
        self.fe = feature_engineering.FeatureEngineeringClient()

    def create_online_table(self):
        """
        Creates an online table based on the feature table.
        """
        spec = OnlineTableSpec(
            primary_key_columns=["player_name"],  # Just player_name since we aggregated
            source_table_full_name=self.feature_table_name,
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict(
                {"triggered": "true"}
            ),
            perform_full_copy=False,
        )
        try:
            existing_table = self.workspace.online_tables.get(self.online_table_name)
            if existing_table:
                self.logger.info(f"Updating existing table: {self.online_table_name}")
                self.workspace.online_tables.update(
                    name=self.online_table_name, spec=spec
                )

            else:
                self.logger.info(f"Creating new table: {self.online_table_name}")
                self.workspace.online_tables.create(
                    name=self.online_table_name, spec=spec
                )

        except Exception as e:
            self.logger.error(f"Error creating online table: {e}")

    def create_feature_spec(self):
        """
        Creates a feature spec to enable feature serving.
        """
        features = [
            FeatureLookup(
                table_name=self.feature_table_name,
                lookup_key="player_name",  # Changed from Id to player_name
                feature_names=[
                    "age",
                    "team_abbreviation",
                    "Predicted_Points",  # Changed from Predicted_SalePrice
                ],
            )
        ]
        self.fe.create_feature_spec(
            name=self.feature_spec_name, features=features, exclude_columns=None
        )

    def deploy_or_update_serving_endpoint(
        self, workload_size: str = "Small", scale_to_zero: bool = True
    ):
        """
        Deploys the feature serving endpoint in Databricks.
        """
        endpoint_exists = any(
            item.name == self.endpoint_name
            for item in self.workspace.serving_endpoints.list()
        )

        served_entities = [
            ServedEntityInput(
                entity_name=self.feature_spec_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
            )
        ]

        if not endpoint_exists:
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=served_entities,
                ),
            )
        else:
            self.workspace.serving_endpoints.update_config(
                name=self.endpoint_name, served_entities=served_entities
            )
