import os
import warnings
import sys
#%%
import pandas as pd
import numpy as np
#%%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn import ensemble
#%%
import mlflow
import mlflow.sklearn

#%%
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#%%
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

#%%
if __name__ == "__main__":
#%%
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    csv_url =\
            'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

    try:
        data = pd.read_csv(csv_url, sep=';')
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e)

    train, test = train_test_split(data)
    #train.describe()

    #%%
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    #%%
    gbmParams = {'n_estimators': int(sys.argv[1]) if len(sys.argv) > 1 else 500,
                 'max_depth': int(sys.argv[2]) if len(sys.argv) > 2 else 4,
                 'min_samples_split': int(sys.argv[3]) if len(sys.argv) > 3 else 2,
                'learning_rate': float(sys.argv[4]) if len(sys.argv) > 4 else 0.01,
                 'loss': sys.argv[5] if len(sys.argv) > 5 else 'ls'}
    #%%
    with mlflow.start_run():
    #%%
        #lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=4233)
        lr = ensemble.GradientBoostingRegressor(**gbmParams)
        lr.fit(train_x, train_y)

        #%%
        predicted_qualities = lr.predict(test_x)

        #%%
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("GBM model (n_estimators=%d, "
              "max_depth=%d, "
              "min_samples_split=%d, "
              "learning_rate=%f, "
              "loss=%s):" % (gbmParams.get('n_estimators'),
                             gbmParams.get('max_depth'),
                             gbmParams.get('min_samples_split'),
                             gbmParams.get('learning_rate'),
                             gbmParams.get('loss') ))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        #%%
        #mlflow.log_param("alpha", alpha)
        #mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_params(gbmParams)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model_gbm")