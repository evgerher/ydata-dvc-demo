import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import catboost as cb
import xgboost as xg
from sklearn.pipeline import Pipeline

from ml_project.entities import ModelParams


def model_factory(params: ModelParams):
    if params.model_type == 'random_forest':
        return random_forest(params.kwargs, params.random_state)
    elif params.model_type == 'linear_regression':
        return create_linear_regression(params.kwargs)
    elif params.model_type == 'xgboost':
        return xgboost_regression(params.kwargs, params.random_state)
    elif params.model_type == 'catboost':
        return catboost_regression(params.kwargs, params.random_state)
    else:
        raise NotImplementedError(f'Model type unsupported [{params.model_type}]')


def save_model(model, metrics: dict, pipeline: Pipeline, output_model_path: str):
    data = {
        'model': model,
        'metrics': metrics,
        'pipeline': pipeline
    }
    with open(output_model_path, 'wb') as f:
        pickle.dump(data, f)


def random_forest(kwargs: dict, random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(**kwargs, random_state=random_state)


def create_linear_regression(kwargs: dict) -> LinearRegression:
    return LinearRegression(**kwargs)


def xgboost_regression(kwargs: dict, random_state: int) -> xg.XGBRegressor:
    return xg.XGBRegressor(**kwargs, random_state=random_state)


def catboost_regression(kwargs: dict, random_state: int) -> cb.CatBoostRegressor:
    return cb.CatBoostRegressor(**kwargs, random_state=random_state)
