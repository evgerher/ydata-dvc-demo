import pickle
from pathlib import Path
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import catboost as cb
import xgboost as xg
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


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
    elif params.model_type == 'random_guesser':
        return random_guesser_regression(params.kwargs, params.random_state)
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
    return cb.CatBoostRegressor(**kwargs, random_state=random_state, verbose=False)


def random_guesser_regression(kwargs: dict, random_state: int) -> 'RandomGuesser':
    low = kwargs.get('low', 10)
    high = kwargs.get('high', 100)
    return RandomGuesser(random_state, low, high)


class RandomGuesser(BaseEstimator, RegressorMixin):
    def __init__(self, random_state: int, low: int = 10, high: int = 100):
        self.random_state = random_state
        random.seed(random_state)
        self.guesser = random.Random(random_state)

        assert low <= high, f'Low value must be less or equal to high value, but found: {low} and {high}'
        self.low = low
        self.high = high


    def fit(self, X: np.array, y: np.array):
        return self

    def predict(self, X: np.array):
        shape = X.shape # B x F, B - batch size, F - features
        N = shape[0]
        outs = []
        for _ in range(N):
            number = self.guesser.randint(self.low, self.high)
            outs.append(number)
        outs = np.array(outs)
        assert len(outs) == len(X), 'Provided batch X and predictions should match in first dimension!'

        return outs
