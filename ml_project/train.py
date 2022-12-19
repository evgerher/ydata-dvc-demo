import pickle
from argparse import ArgumentParser
from pathlib import Path
import logging

from ml_project.data import read_data, split_train_val_data, download_data
from ml_project.entities import read_training_pipeline_params, TrainingPipelineParams, SplittingParams, FeatureParams
from ml_project.features import featurize, prepare_pipeline
from ml_project.models import model_factory, compute_metrics, save_model
from ml_project.utils.helpers import setup_logger

logger = logging.getLogger('trainer')

params: TrainingPipelineParams = read_training_pipeline_params('configs/train.yaml')

def data_callback(args):
    output_dir = Path(args.output_path)
    download_data(args.train_path, output_dir / 'train.csv')
    download_data(args.test_path, output_dir / 'test.csv')


def featurizer_callback(args):
    logger.info(f'Start featurize, feature_params={params.feature_params}; split_params={params.splitting_params}')
    data_folder = Path(args.data_folder)
    train_path = data_folder / 'train.csv'
    test_path = data_folder / 'test.csv'
    ds_train, ds_test = read_data(train_path), read_data(test_path)
    logger.info(f'Loaded datasets from {data_folder.absolute()}')

    output_folder = Path(args.output_folder)

    transformer = prepare_pipeline(params.feature_params)
    transformer.fit(ds_train)
    ds_train = featurize(ds_train, transformer, target_column=params.feature_params.target_col)
    ds_test = featurize(ds_test, transformer, target_column=None)
    ds_train, ds_val = split_train_val_data(ds_train, params.splitting_params)
    logger.info(f'Featurized and splitted dataset')

    with open(output_folder / 'train.pkl', 'wb') as train_f, \
            open(output_folder / 'val.pkl', 'wb') as val_f, \
            open(output_folder / 'test.pkl', 'wb') as test_f, \
            open(output_folder / 'transformer.pkl', 'wb') as transformer_f:
        pickle.dump(ds_train, train_f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(ds_val, val_f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(ds_test, test_f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(transformer, transformer_f, protocol=pickle.HIGHEST_PROTOCOL)


def model_callback(args):
    pass


def parse_args():
    parser = ArgumentParser(
        prog='bike-sharing',
        description='Regressor trainer for bike sharing task'
    )

    subparsers = parser.add_subparsers(help='Choose command to use')

    data_parser = subparsers.add_parser('load_data', help="Load data from remote")
    data_parser.add_argument('--train_cloud_path',
                             type=str,
                             required=True,
                             dest='train_path',
                             metavar='path/to/data.csv',
                             help='Url to data file')
    data_parser.add_argument('--test_cloud_path',
                             type=str,
                             required=True,
                             dest='test_path',
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
    model_parser.set_defaults(callback=train_model_callback)
    return parser.parse_args()


def load_ds(folder: Path):
    with open(folder / 'train.pkl', 'rb') as train_f, \
            open(folder / 'val.pkl', 'rb') as val_f, \
            open(folder / 'test.pkl', 'rb') as test_f, \
            open(folder / 'transformer.pkl', 'rb') as transformer_f:
        ds_train = pickle.load(train_f)
        ds_val = pickle.load(val_f)
        ds_test = pickle.load(test_f)
        transformer = pickle.load(transformer_f)
        return (ds_train, ds_val, ds_test), transformer


def train_model_callback(args):
    logger.info(f'Starting training procedure, configs={params.train_params}')
    data_folder = Path(args.data_folder)
    (ds_train, ds_val, ds_test), transformer = load_ds(data_folder)
    model = model_factory(params.train_params)
    model.fit(ds_train.X, ds_train.Y)
    model_predictions = model.predict(ds_val.X)
    metrics = compute_metrics(ds_val.Y, model_predictions)
    logger.info(f'Model performance: {metrics}')
    save_model(model, metrics, transformer, params.output_model_path)

    test_predictions = model.predict(ds_test.X)
    with open(params.test_predictions_path, 'wb') as f:
        pickle.dump(test_predictions, f)

    logger.info('Finished training procedure...')


if __name__ == '__main__':
    setup_logger('trainer')
    arguments = parse_args()
    arguments.callback(arguments)
