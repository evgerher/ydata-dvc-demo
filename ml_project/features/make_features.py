from dataclasses import dataclass, field
from typing import Optional, List

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, FunctionTransformer

from ml_project.entities import FeatureParams


@dataclass
class Dataset:
    X: pd.DataFrame
    Y: Optional[pd.DataFrame] = field(default=None)


def split_datetime(df: pd.DataFrame, date_column: str):
    dt_column = df[date_column]
    seconds = dt_column.second
    minutes = dt_column.minute
    hours = dt_column.hour
    days = dt_column.day
    monthes = dt_column.month
    years = dt_column.year
    weekdays = dt_column.dayofweek
    return (seconds, minutes, hours, days, monthes, years, weekdays)


class DatetimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str]):
        self.columns = columns

    def _check_columns(self, columns: List[str]):
        assert all(col in columns for col in self.columns), \
            f'Required columns not found in column list, required={self.columns}, found={columns}'

    def fit(self, X: pd.DataFrame, y=None):
        self._check_columns(X.columns)
        return self

    def transform(self, X, y=None):
        self._check_columns(X.columns)
        for prefix in self.columns:
            X[prefix] = pd.to_datetime(X[prefix])
            new_fields = ['second', 'minute', 'hour', 'day', 'month', 'year', 'weekday']
            new_column_names = [f'{prefix}_{col_name}' for col_name in new_fields]
            X[new_column_names] = X.apply(split_datetime, date_column=prefix, axis=1, result_type="expand")
            X.drop(columns=[prefix], inplace=True)
        return X


def make_transformer_numerical(policy: str) -> Pipeline:
    if policy == 'standard_scaler':
        pipe = [(policy, StandardScaler())]
    elif policy == 'identity':
        pipe = [(policy, FunctionTransformer())]
    else:
        raise NotImplementedError(f'Unknown policy: {policy}')
    return Pipeline(pipe)


def make_transformer_date(date_columns: List[str]) -> Pipeline:
    pipe = [('date_expander', DatetimeTransformer(date_columns))]
    return Pipeline(pipe)


class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        for col in X.columns:
            encoder = LabelEncoder()
            encoder.fit(X[col])
            self.encoders[col] = encoder
        return self

    def transform(self, X, y=None):
        for col in X.columns:
            data = X[col]
            X[col] = self.encoders[col].transform(data)

        return X


def make_transformer_categorical(policy: str) -> Pipeline:
    if policy == 'ohe': # one hot encoding
        pipe = [(policy, OneHotEncoder())]
    elif policy == 'label_encoder':
        pipe = [(policy, MultiColumnLabelEncoder())]
    else:
        raise NotImplementedError(f'Unknown policy: {policy}')
    return Pipeline(pipe)


def prepare_pipeline(params: FeatureParams):
    column_converters = [
        ('categorical', make_transformer_categorical(params.categorical_policy), params.categorical_features),
        ('numerical', make_transformer_numerical(params.numerical_policy), params.numerical_features),
        ('date', make_transformer_date([params.date_column]), [params.date_column])
    ]

    transformer = ColumnTransformer(column_converters)
    if params.extra_features_enabled:
        pipe = Pipeline([
            ('extra_featurizer', ExtraFeaturesTransformer()),
            ('column_transformer', transformer)
        ])
        return pipe
    return transformer


class ExtraFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.columns_required = {'atemp', 'temp', 'windspeed'}
        self.mean_values = {}

    def _check_columns(self, columns: List[str]):
        assert all(col in columns for col in self.columns_required), \
            f'Required columns not found in column list, required={self.columns_required}, found={columns}'

    def fit(self, X: pd.DataFrame, y=None):
        self._check_columns(X.columns)

        mean_delta = (X['atemp'] - X['temp']).mean()
        mean_windspeed = X['windspeed'].mean()
        self.mean_values['delta_atemp_temp'] = mean_delta
        self.mean_values['windspeed'] = mean_windspeed

        return self

    def transform(self, X, y=None):
        self._check_columns(X.columns)

        X['delta_temp_atemp'] = X['atemp'] - X['temp']
        X['is_high_delta'] = X['delta_temp_atemp'] > self.mean_values['delta_atemp_temp']

        X['is_windy'] = X['windspeed'] > self.mean_values['windspeed']
        X['delta_and_wind'] = X['is_high_delta'] & X['is_windy']
        return X


def featurize(df: pd.DataFrame,
              transformer: Pipeline,
              target_column: Optional[str] = None) -> Dataset:
    X = pd.DataFrame(transformer.transform(df))
    Y = None
    if target_column is not None:
        Y = df[target_column]

    return Dataset(X=X, Y=Y)
