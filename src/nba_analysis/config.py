from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    """Project Configuration"""

    data: str
    id_column: str
    num_features: List[str]
    cat_features: List[str]
    target: str
    features: List[str]
    catalog_name: str
    schema_name: str
    experiment_name: str
    experiment_name_fe: str
    endpoint_name: str
    parameters: Dict[str, Any]
    pipeline_id: Optional[str]
    packages: Optional[List[str]]

    @classmethod
    def from_yaml(cls, config_path: str, env: str = None):
        """Load config from YAML file"""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        if env is not None:
            # Load environment-specific settings
            config_dict["catalog_name"] = config_dict[env]["catalog_name"]
            config_dict["schema_name"] = config_dict[env]["schema_name"]
            config_dict["pipeline_id"] = config_dict[env]["pipeline_id"]
            config_dict["data"] = config_dict[env]["data"]
        # Default case when no environment is specified should also work

        return cls(**config_dict)


# For backward compatibility, create an alias for Config
ProjectConfig = Config


class Tags(BaseModel):
    git_sha: str
    branch: str
    job_run_id: Optional[str] = None
