from dataclasses import dataclass, field
from typing import List


@dataclass
class FeatureParams:
    date_column: str
    categorical_features: List[str]
    categorical_policy: str
    numerical_features: List[str]
    numerical_policy: str
    target_col: str
    extra_features_enabled: bool = field(default=False)
