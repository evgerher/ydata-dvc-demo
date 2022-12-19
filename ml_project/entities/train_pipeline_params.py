from dataclasses import dataclass
from pathlib import Path

from .dataset_params import SplittingParams
from .feature_params import FeatureParams
from .model_params import ModelParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: ModelParams
    test_predictions_path: str


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: Path) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
