import logging
from pathlib import Path

import pandas as pd
# from joblib import dump
# from sklearn import preprocessing
# from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# from sklearn.ensemble import HistGradientBoostingRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml

logger = logging.getLogger(__name__)

# def prepare_dataset(test_size=0.2, random_seed=1):
#     dataset = pd.read_csv(
#         "winequality-red.csv",
#         delimiter=",",
#     )
#     dataset = dataset.rename(columns=lambda x: x.lower().replace(" ", "_"))
#     train_df, test_df = train_test_split(dataset, test_size=test_size, 
#                         random_state=random_seed)
#     return {"train": train_df, "test": test_df}

# def train():
#     logger.info("Preparing dataset...")
#     dataset = prepare_dataset()
#     train_df = dataset["train"]
#     test_df = dataset["test"]

#     # separate features from target
#     y_train = train_df["quality"]
#     X_train = train_df.drop("quality", axis=1)
#     y_test = test_df["quality"]
#     X_test = test_df.drop("quality", axis=1)

#     logger.info("Training model...")
#     scaler = preprocessing.StandardScaler().fit(X_train)
#     X_train = scaler.transform(X_train)
#     X_test = scaler.transform(X_test)
#     model = HistGradientBoostingRegressor(max_iter=50).fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     error = mean_squared_error(y_test, y_pred)
#     logger.info(f"Test MSE: {error}")

#     logger.info("Saving artifacts...")
#     Path("artifacts").mkdir(exist_ok=True)
#     dump(model, "artifacts/model.joblib")
#     dump(scaler, "artifacts/scaler.joblib")

def train():
    logger.info("Preparing dataset...")

    iris_df = pd.read_csv("Iris.csv")

    # separate features from target
    iris_y = iris_df["Species"]
    iris_X = iris_df.drop(["Id", "Species"], axis=1)

    logger.info("Training model...")
    pipeline = PMMLPipeline([
        ("classifier", DecisionTreeClassifier())
    ])
    pipeline.fit(iris_X, iris_y)

    logger.info("Saving artifacts...")
    sklearn2pmml(pipeline, "DecisionTreeIris.pmml", with_repr = True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()