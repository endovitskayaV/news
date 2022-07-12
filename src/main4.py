import ast
import logging
from typing import List, Dict

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.ature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from settings import RAW_PATH, LOGGING_PATH, DATA_PATH
from src.utils import identity, dump, write_to_file


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


logger = logging.getLogger()
formatter = logging.Formatter(
    "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
)
general_fh = logging.FileHandler(LOGGING_PATH / "logs4.txt")
general_fh.setFormatter(formatter)
general_fh.setLevel("INFO")
logger.addHandler(general_fh)


def str_to_list(row: Series, col_name: str) -> Series:
    row[col_name] = ast.literal_eval(row[col_name])
    return row


df_train = pd.read_csv(DATA_PATH / "df_text.csv")
df_train = df_train.apply(lambda row: str_to_list(row, 'title'), axis=1)
df_train = df_train[df_train['views'] > 1_000_000]
df_train = df_train[df_train['depth'] > 1.7]
df_train = df_train[df_train['full_reads_percent'] > 100]

full_df_train = pd.read_csv(RAW_PATH / "train.csv", index_col=0)
full_df_train = full_df_train.reset_index()
full_df_train = encode_dummies(full_df_train, 'category')
full_df_train = encode_list_by_rate(full_df_train, 'authors', 0.03)
full_df_train = encode_list_by_rate(full_df_train, 'tags', 0.01)
full_df_train['day'] = pd.to_datetime(full_df_train['publish_date']).dt.strftime("%d").astype(int)
full_df_train['month'] = pd.to_datetime(full_df_train['publish_date']).dt.strftime("%m").astype(int)
full_df_train['text'] = df_train['title']

X = full_df_train.drop(
    ["views", "depth", "full_reads_percent", "title", "publish_date", "session", "document_id"], axis=1)
y = full_df_train[["views", "depth", "full_reads_percent"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

vectorizer = TfidfVectorizer(tokenizer=identity, lowercase=False)
train_texts = vectorizer.fit_transform(X_train['text'])
test_texts = vectorizer.transform(X_test['text'])
dump(DATA_PATH / "vectorizer23.pickle", vectorizer)

train_texts_arr = train_texts.toarray()
train_texts_df = pd.DataFrame(train_texts_arr)
test_texts_df = pd.DataFrame(test_texts.toarray())

train_texts_df.rename(lambda col_name: "text_" + str(col_name), axis='columns', inplace=True)
test_texts_df.rename(lambda col_name: "text_" + str(col_name), axis='columns', inplace=True)

X_train = X_train.reset_index()
X_test = X_test.reset_index()
y_train = y_train.reset_index()
y_test = y_test.reset_index()

X_train = X_train.merge(train_texts_df, left_index=True, right_index=True)
X_test = X_test.merge(test_texts_df, left_index=True, right_index=True)
X_train.drop(['text'], axis=1, inplace=True)
X_test.drop(['text'], axis=1, inplace=True)

feature_array = np.array(vectorizer.get_feature_names_out())
tfidf_sorting = np.argsort(train_texts_arr).flatten()[::-1]
n = 500
top_n = feature_array[tfidf_sorting][:n]
write_to_file(DATA_PATH / "top500.txt", '\n'.join(p for p in top_n))

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


def train_score(y_cols: List[str], X_train, X_test, y_train, y_test):
    y_train = y_train[y_cols]
    y_test = y_test[y_cols]

    if y_train.shape[1] == 1:
        y_train = y_train.values.ravel()

    logger.log(msg="y_cols " + str(y_cols), level=logging.getLevelName("WARNING"))

    random_grid = {
        "n_estimators": [200, 500, ],
        'max_features': [0.6, 0.8, 1],
        "max_depth": [20, 50, ],
    }

    rf_random = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=random_grid,
                                   n_iter=100, scoring='r2',
                                   cv=3, verbose=2, random_state=42, n_jobs=-1,
                                   return_train_score=True)
    search = rf_random.fit(X_train, y_train)
    dump(DATA_PATH / "search.pickle", search)
    logger.log(msg="params " + str(search.best_params_), level=logging.getLevelName("WARNING"))
    logger.log(msg="best_score_ " + str(search.best_score_), level=logging.getLevelName("WARNING"))
    logger.log(msg="train score r2 " + str(r2_score(y_train, search.predict(X_train))),
               level=logging.getLevelName("WARNING"))
    logger.log(msg="train score " + str(search.score(X_train, y_train)), level=logging.getLevelName("WARNING"))
    logger.log(msg="test score " + str(search.score(X_test, y_test)), level=logging.getLevelName("WARNING"))
    logger.log(msg="test score r2 " + str(r2_score(y_test, search.predict(X_test))),
               level=logging.getLevelName("WARNING"))
    score = calculate_score(y_test, search.predict(X_test), y_cols)
    logger.log(msg="test score custom " + str(score), level=logging.getLevelName("WARNING"))

    # params_grid=[
    #     {'n_estimators':200, 'max_depth':20, 'min_samples_split':2, 'max_features':1.0, 'bootstrap':True, 'n_jobs':-1, 'max_samples':None},
    #     {'n_estimators':500, 'max_depth':20, 'min_samples_split':2, 'max_features':1.0, 'bootstrap':True, 'n_jobs':-1, 'max_samples':None},
    #     {'n_estimators':1000, 'max_depth':20, 'min_samples_split':2, 'max_features':1.0, 'bootstrap':True, 'n_jobs':-1, 'max_samples':None},
    #     {'n_estimators':200, 'max_depth':100, 'min_samples_split':2, 'max_features':1.0, 'bootstrap':True, 'n_jobs':-1, 'max_samples':None},
    #     {'n_estimators':200, 'max_depth':200, 'min_samples_split':2, 'max_features':1.0, 'bootstrap':True, 'n_jobs':-1, 'max_samples':None},
    #     {'n_estimators':200, 'max_depth':20, 'min_samples_split':2, 'max_features':5000, 'bootstrap':True, 'n_jobs':-1, 'max_samples':None},
    #     {'n_estimators':200, 'max_depth':20, 'min_samples_split':2, 'max_features':2500, 'bootstrap':True, 'n_jobs':-1, 'max_samples':None},
    #     {'n_estimators':100, 'max_depth':20, 'min_samples_split':2, 'max_features':1000, 'bootstrap':True, 'n_jobs':-1, 'max_samples':None},
    #     {'n_estimators':200, 'max_depth':20, 'min_samples_split':2, 'max_features':1500, 'bootstrap':True, 'n_jobs':-1, 'max_samples':None},
    # ]
    # for params in params_grid:
    #     logger.log(msg="params " + str(params), level=logging.getLevelName("WARNING"))
    #     regr = RandomForestRegressor(**params)
    #     regr.fit(X_train, y_train)
    #
    #     pred = regr.predict(X_test)
    #     logger.log(msg="train score " + str(regr.score(X_train, y_train)), level=logging.getLevelName("WARNING"))
    #     logger.log(msg="train score r2 " + str(r2_score(y_train, regr.predict(X_train))), level=logging.getLevelName("WARNING"))
    #     logger.log(msg="test score r2 " + str(r2_score(y_test, pred)), level=logging.getLevelName("WARNING"))
    #     score = calculate_score(y_test, pred, y_cols)
    #     logger.log(msg="test score "+str(score), level=logging.getLevelName("WARNING"))


for y_cols in ([["views"], ["depth"], ["full_reads_percent"], ["views", "depth", "full_reads_percent"]]):
    train_score(y_cols, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
