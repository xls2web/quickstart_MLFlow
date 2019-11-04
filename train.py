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
    #alpha = 0.5
    #l1_ratio = 0.5
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    #%%
    with mlflow.start_run():
    #%%
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=4233)
        lr.fit(train_x, train_y)

        #%%
        predicted_qualities = lr.predict(test_x)

        #%%
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        #%%
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")
