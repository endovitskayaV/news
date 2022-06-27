import ast
from typing import Dict, List

import pandas as pd
from pandas import DataFrame
from pandas import Series
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import logging

from settings import RAW_PATH, LOGGING_PATH

logger = logging.getLogger("ll")
formatter = logging.Formatter(
    "%(asctime)s %(name)-12s %(message)s"
)
general_fh = logging.FileHandler(LOGGING_PATH / "logs.txt")
general_fh.setFormatter(formatter)
logger.addHandler(general_fh)

df_train = pd.read_csv(RAW_PATH / "train.csv", index_col=0)
df_test = pd.read_csv(RAW_PATH / "test.csv", index_col=0)


def encode_dummies(df: DataFrame, col_name: str) -> DataFrame:
    categories_df = pd.get_dummies(df[col_name], prefix=col_name)  # dummy_na = False -> no NaN found
    df = df.drop(col_name, axis=1)
    df = df.join(categories_df)
    return df


def encode_list_by_rate(df: DataFrame, col_name: str, rate_limit: float) -> DataFrame:
    def str_to_list(row: Series, col_name: str) -> Series:
        row[col_name] = ast.literal_eval(row[col_name])
        return row

    def get_col_encode_dict(df: DataFrame, col_name: str, rate_limit: float) -> Dict[str, int]:
        col_value_rates = df.explode(col_name)[col_name].value_counts(normalize=True)
        col_encode_dict = {}
        for index, (col_value, rate) in enumerate(col_value_rates.items()):
            if rate < rate_limit:
                break
            col_encode_dict[col_value] = index * 10

        return col_encode_dict

    def encode(row: Series, col_name: str, encode_dict: Dict[str, int], empty_code: int) -> Series:
        values = row[col_name]
        code = 0
        if len(values) == 0:
            code = empty_code
        else:
            for col_value, col_code in encode_dict.items():
                if col_value in values:
                    code += col_code

        row[col_name] = code
        return row

    df = df.apply(lambda row: str_to_list(row, col_name), axis=1)
    col_encode_dict = get_col_encode_dict(df, col_name, rate_limit)
    df = df.apply(lambda row: encode(row, col_name, col_encode_dict, -1), axis=1)
    return df


df_train = encode_dummies(df_train, 'category')
df_train = encode_list_by_rate(df_train, 'authors', 0.03)
df_train['day'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%d").astype(int)
df_train['month'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%m").astype(int)

score_dict = {"views": 0.4, "depth": 0.3, "full_reads_percent": 0.3}


def calculate_score(y_true: Series, y_pred: Series, y_cols: List[str]) -> float:
    score = 0
    for i, col_name in enumerate(y_cols):
        if len(y_cols) > 1:
            y_pred_i = y_pred[:, i]
            y_true_i = y_true[col_name]
            score_coef = score_dict[col_name]
        else:
            y_pred_i = y_pred
            y_true_i = y_true[col_name].ravel()
            score_coef = 1
        score += score_coef * r2_score(y_true_i, y_pred_i)
    return score


X = df_train.drop(["views", "depth", "full_reads_percent", "title", "publish_date", "session", "tags"], axis=1)
y = df_train[["views", "depth", "full_reads_percent"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def train_score(y_cols: List[str], X_train, X_test, y_train, y_test):
    y_train = y_train[y_cols]
    y_test = y_test[y_cols]

    if y_train.shape[1] == 1:
        y_train = y_train.values.ravel()

    estimator = RandomForestRegressor()

    param_grid = {
        "n_estimators": [100, 500, 1000],
        "max_features": ["auto", "sqrt", "log2"],
        "min_samples_split": [2, 4, 5],
        "bootstrap": [True, False],
    }
    grid = GridSearchCV(estimator, param_grid, n_jobs=-1, cv=5)
    grid.fit(X_train, y_train)
    logger.log(msg="y_cols " + str(y_cols), level=logging.getLevelName("WARNING"))
    logger.log(msg="best_score " + str(grid.best_score_), level=logging.getLevelName("WARNING"))
    logger.log(msg="best_params " + str(grid.best_params_), level=logging.getLevelName("WARNING"))

    regr = RandomForestRegressor(**grid.best_params_)
    regr.fit(X_train, y_train)

    pred = regr.predict(X_test)
    score = calculate_score(y_test, pred, y_cols)
    logger.log(msg="score " + str(score), level=logging.getLevelName("WARNING"))

    col_name = 'importance'
    importance_df = pd.DataFrame(regr.feature_importances_, columns=[col_name],
                                 index=regr.feature_names_in_).sort_values(by=col_name, ascending=False)
    logger.log(msg="importance_df " + str(importance_df), level=logging.getLevelName("WARNING"))


for y_cols in ([["views", "depth", "full_reads_percent"]]):
    train_score(y_cols, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
