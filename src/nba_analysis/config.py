# config.py
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    # Core configuration fields without default values
    input_data: str
    local_data: str = "data/raw/all_seasons.csv"  
    experiment_name: str
    experiment_name_fe: str 
    catalog_name: str    
    schema_name: str
    target: str
    num_features: List[str]
    cat_features: List[str]
    parameters: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class Tags(BaseModel):
    git_sha: str
    branch: str
