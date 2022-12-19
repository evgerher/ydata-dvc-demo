import pickle
from argparse import ArgumentParser
from pathlib import Path
import logging

from ml_project.data import read_data, split_train_val_data
from ml_project.entities import read_training_pipeline_params, TrainingPipelineParams, SplittingParams, FeatureParams
from ml_project.features import featurize, prepare_pipeline
from ml_project.models import model_factory, compute_metrics, save_model
from ml_project.utils.helpers import setup_logger

logger = logging.getLogger('trainer')


def data_callback(args):
    pass


def featurizer_callback(args):
    pass


def model_callback(args):
    pass


def parse_args():
    parser = ArgumentParser(
        prog='bike-sharing',
        description='Regressor trainer for bike sharing task'
    )

    subparsers = parser.add_subparsers(help='Choose command to use')

    data_parser = subparsers.add_parser('load_data', help="Load data from remote")

    data_parser.add_argument('--train_url',
                             type=str,
                             required=True,
                             metavar='path/to/data.csv',
                             help='Url to data file')
    data_parser.add_argument('--test_url',
                             type=str,
                             required=True,
                             metavar='path/to/data.csv',
                             help='Url to data file')
    data_parser.add_argument('--output_path',
                             type=str,
                             default='data/raw/',
                             help='Path to store downloaded data')
    data_parser.set_defaults(callback=data_callback)

    features_parser = subparsers.add_parser('featurize', help="Featurize loaded datasets")

    features_parser.add_argument('--data_folder',
                                 type=str,
                                 default='data/raw',
                                 help='Path to stored data')
    features_parser.add_argument('--output_folder',
                                 default='data/processed',
                                 help='Path to save results')
    features_parser.set_defaults(callback=featurizer_callback)

    model_parser = subparsers.add_parser('train_model', help='Train model')
    model_parser.add_argument('--data_folder',
                                 type=str,
                                 default='data/processed',
                                 help='Path to stored data')
    model_parser.add_argument('--output_model_path',
                                 type=str,
                                 default='artifacts/model.pkl',
                                 help='Path to save result model')
    model_parser.set_defaults(callback=model_callback)

    parser.add_argument('--config',
                        type=str,
                        dest='config_path',
                        required=True,
                        metavar='config/train.yaml',
                        help='Path to config file to run train with')
    return parser.parse_args()


def prepare_datasets(data_path: str,
                     split_params: SplittingParams,
                     feature_params: FeatureParams):
    data_path = Path(data_path)
    train_path = data_path / 'train.csv'
    test_path = data_path / 'test.csv'
    ds_train, ds_test = read_data(train_path), read_data(test_path)
    logger.info(f'Loaded datasets from {[train_path.absolute(), test_path.absolute()]}')
    transformer = prepare_pipeline(feature_params)
    transformer.fit(ds_train)
    ds_train = featurize(ds_train, transformer, target_column=feature_params.target_col)
    ds_test = featurize(ds_test, transformer, target_column=None)
    ds_train, ds_val = split_train_val_data(ds_train, split_params)
    logger.info(f'Featurized and splitted dataset')

    return (ds_train, ds_val, ds_test), transformer


def train_model(args):
    config: TrainingPipelineParams = read_training_pipeline_params(args.config_path)
    logger.info(f'Starting training procedure, configs={config}')
    (ds_train, ds_val, ds_test), transformer = prepare_datasets(config.input_data_path,
                                                                config.splitting_params,
                                                                config.feature_params)
    model = model_factory(config.train_params)
    model.fit(ds_train.X, ds_train.Y)
    model_predictions = model.predict(ds_val.X)
    metrics = compute_metrics(ds_val.Y, model_predictions)
    logger.info(f'Model performance: {metrics}')
    save_model(model, metrics, transformer, config.output_model_path)

    test_predictions = model.predict(ds_test.X)
    with open(config.test_predictions_path, 'wb') as f:
        pickle.dump(test_predictions, f)

    logger.info('Finished training procedure...')


if __name__ == '__main__':
    setup_logger('trainer')
    arguments = parse_args()
    arguments.callback(arguments)
