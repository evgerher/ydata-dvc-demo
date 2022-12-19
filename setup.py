from setuptools import setup, find_packages

setup(
  name='ml-project',
  version='0.0.1',
  packages=find_packages(include=['ml_project', 'ml_project.*']),
  url='',
  license='',
  author='Evgenii Sorokin',
  author_email='evgerher@toloka.ai',
  description='ML Project example to demonstrate tools [s3, dvc, mlflow] and how to transform jupyter notebook into code'
)