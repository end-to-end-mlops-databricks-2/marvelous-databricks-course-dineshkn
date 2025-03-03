import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput


class ModelServing:
    def __init__(self, model_name: str, endpoint_name: str):
        """
        Initializes the Model Serving Manager.
        """
        self.workspace = WorkspaceClient()
        self.endpoint_name = endpoint_name
        self.model_name = model_name

    def get_latest_model_version(self):
        """Get the latest version of the model."""
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(
            self.model_name, alias="latest-model"
        ).version

        latest_version = int(latest_version)
        print(f"Latest model version: {latest_version} (Type: {type(latest_version)})")
        return latest_version

    def deploy_or_update_serving_endpoint(
        self,
        version: str = "latest",
        workload_size: str = "Small",
        scale_to_zero: bool = True,
    ):
        """
        Deploys the model serving endpoint in Databricks.

        Args:
            version: Version of the model to deploy
            workload_size: Workload size (number of concurrent requests)
            scale_to_zero: If True, endpoint scales to 0 when unused
        """
        endpoint_exists = any(
            item.name == self.endpoint_name
            for item in self.workspace.serving_endpoints.list()
        )

        if version == "latest":
            entity_version = self.get_latest_model_version()
        else:
            entity_version = version

        print(f"Entity version: {entity_version}, Type: {type(entity_version)}")

        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
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
