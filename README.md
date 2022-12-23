# ML project example (ydata-demo)

### Dependencies installation

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Usage

Load data `python ml_project/train.py load_data --train_cloud_path dvc/train.csv --test_cloud_path dvc/test.csv`     
Featurize `python ml_project/train.py featurize --data_folder data/raw --output_folder data/processed`  
Train model `python ml_project/train.py train_model --data_folder data/processed`  

## DVC

Examples
```
dvc add /path/to/item.pkl
dvc push
dvc repro
```

### DVC initialization

```
dvc init
dvc remote add s3cache s3://evgerher-ydata-demo/dvc
dvc remote modify s3cache profile ydata-demo
dvc remote modify s3cache endpointurl https://storage.yandexcloud.net
```

Optional: `dvc remote default s3cache`


## MlFlow

Resources:  
- https://www.mlflow.org/docs/latest/quickstart.html  
- https://www.mlflow.org/docs/latest/tracking.html#concepts  

Automatic parameters tracking examples:  
- https://www.mlflow.org/docs/latest/python_api/mlflow.catboost.html  
- https://www.mlflow.org/docs/latest/python_api/mlflow.pytorch.html  

---

### Project structure

```
- ml_project - folder with source code
  - data: package for loading from local/remote source data
  - features: package for data transformation
  - models: package for model training procedures
- configs - folder with configuration files to run training experiments
  - train_lr.yaml - example of linear regression config
  - train_rf.yaml - example of random forest regressor config
- tests - folder with tests files
- notebooks - folder with initial .ipynb files
- artifacts
  - models - trained model binary file and other files

requirements.txt   <- The requirements file for reproducing the analysis environment
```
