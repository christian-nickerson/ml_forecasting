# Stock Price Prediction

An application that explores the performance of different machine learning algorithims for prediction a given stocks price over time.

1. [Overview](#overview)
2. [Quick Start](#quick-start)
    - [Running as a container](#running-as-a-container)
    - [Running as a python application](#running-as-a-python-application-linux--mac)
3. [Configuration](#configuration)
4. [Artifacts & Reports](#artifacts-and-reports)

## Overview

The application takes a given stock (as avaialble through yahoo finance's API), models and parameter tunes the given machine learning model before producing a plotly report on the models trianing and test set predictive performance. **Disclaimer:** For *production* grade predictions, far more focus would be given to better feature engineering, including the inclusion of company financials, news & social sentiment & other statistically similar stocks.

Curretnly the project supports the following machine learning algorthims:

- XGBoost
- LSTM Network (Tensorflow)

The project is not designed to create production grade predictions of a given stock price over time, rather, the project tries to explore the heuristic performance of different models on a given forecsting problem. 

## Quick start

Clone the project into a local directory.

``` bash
git clone https://github.com/christian-nickerson/stock_price_prediction
```

The application can either be run a locally run python application OR as a docker container.

### Running as a container

The run as a container, check docker is correctly installed and running (if not, please follow the docker guide [here](https://docs.docker.com/get-docker/)):

``` bash
docker container run hello-world
```

If "Hello from Docker" is returned correctly, the the image can be built. To build the application image, run the following:

``` bash
docker build . --no-cache -t stock_price_prediction:latest -f src/Dockerfile
```

Once built, the application can be run using the following:

``` bash
docker run --rm -e STOCK_SYMBOL="TSLA" -e MODEL_NAME="xgboost" -e DATA_YEARS="5" -e PARAM_SAMPLES="50" stock_price_prediction:latest
```

### Running as a python application (Linux & Mac)

To run as a local python applicaton, ensure Python 3.9 and PIP is isntalled:

``` bash
python --version
pip --version
```

Create and activate a virtual environment:

``` bash
python3 -m venv env
source env/bin/activate
```

Install the application dependencies:

``` bash
pip install -r requirements.txt
```

To configure the application, create a local `.env` file with the following variables:

```
STOCK_SYMBOL="TSLA"
MODEL_NAME="xgboost"
DATA_YEARS="5"
PARAM_SAMPLES="50"
```

To run the application, using the configuration variables, run:

``` bash
python src/main.py
```

## Configuration

The application uses environment variables to configure itself. These are either passed with the docker run command or as a seperate `.env` that is immported at run time. A breakdown of the varaibles is as follows:

| Variable | Decription |
| --- | --- | 
| STOCK_SYMBOL | Stock symbol to run price prediction on (must match Yahoo Finance symbols) |
| MODEL_NAME | Name of model to train |
| DATA_YEARS | Number of years of data to train model - can cause high memory usage if a large number of years is used |
| PARAM_SAMPLES | Number of samples from paramter space to use for hyperparamter tuning. The greater the number, the more memory required and the longer the train time |

Please be aware, the application is set to utilise as much compute resource as is available locally / provided to the container. Given the intensity of machine learning, this may cause compute and memeory pressure and potentially crash other applications running concurrently.

## Artifacts and Reports

The application has the 3 main folders:

- src: source code for training models
- artifacts: pre-trained model artifact library
- reports: pre-trained model perfromance reports

The application, once a model has been trained, will save the model into the artifact library and save the displayed report in the reports folder. Any new model will overwrite an exisitng model if already contained in the artifact library. **Warning**: if running as a container, ensure the internal artifact and reports folder are mounted to a local directory at runtime to ensure the model and report are saved.

There are a number of pre-trained models as saved examples as part of the application in each of there folders.