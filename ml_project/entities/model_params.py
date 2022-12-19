from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ModelParams:
    model_type: str = field(default="RandomForestRegressor")
    random_state: int = field(default=255)
    kwargs: Dict[str, Any] = field(default_factory=dict)
