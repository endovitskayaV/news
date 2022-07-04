import ast
import logging
import re
import time
from random import randrange
from typing import Dict, List

import numpy as np
import pandas as pd
import spacy_stanza
import stanza
from pandas import DataFrame
from pandas import Series
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from spacy.lang.ru.stop_words import STOP_WORDS

from settings import RAW_PATH, LOGGING_PATH, DATA_PATH
from src.utils import identity, dump, write_to_file, loads

logger = logging.getLogger()
formatter = logging.Formatter(
    "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
)
general_fh = logging.FileHandler(LOGGING_PATH / "logs23.txt")
general_fh.setFormatter(formatter)
general_fh.setLevel("INFO")
logger.addHandler(general_fh)

def str_to_list(row: Series, col_name: str) -> Series:
      row[col_name] = ast.literal_eval(row[col_name])
      return row


df_train = pd.read_csv(DATA_PATH/"df_text.csv")
df_train = df_train.apply(lambda row: str_to_list(row, 'title'), axis=1)

X=df_train
y = df_train[["views", "depth", "full_reads_percent"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_train = X_train['title']
X_test= X_test['title']

vectorizer = TfidfVectorizer(tokenizer=identity, lowercase=False)
X_train = vectorizer.fit_transform(X_train)
X_test= vectorizer.transform(X_test)
dump(DATA_PATH / "vectorizer23.pickle", vectorizer)

feature_array = np.array(vectorizer.get_feature_names_out())
tfidf_sorting = np.argsort(X_train.toarray()).flatten()[::-1]
n = 500
top_n = feature_array[tfidf_sorting][:n]
write_to_file(DATA_PATH / "top500.txt", '\n'.join(p for p in top_n))


score_dict = {"views":0.4, "depth":0.3,"full_reads_percent":0.3}

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

    logger.log(msg="y_cols "+str(y_cols), level=logging.getLevelName("WARNING"))


    random_grid = {
        "n_estimators": [200, 500,],
        'max_features': [0.8],
        "max_depth": [20,50,],
        'min_samples_leaf': [4, 6]

    }

    rf_random = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=random_grid,
                                   n_iter=100, scoring='r2',
                                   cv=3, verbose=2, random_state=42, n_jobs=-1,
                                   return_train_score=True)
    search = rf_random.fit(X_train, y_train)
    dump(DATA_PATH / "search.pickle", search)
    logger.log(msg="params " + str(search.best_params_), level=logging.getLevelName("WARNING"))
    logger.log(msg="best_score_ " + str(search.best_score_), level=logging.getLevelName("WARNING"))
    logger.log(msg="train score r2 " + str(r2_score(y_train, search.predict(X_train))), level=logging.getLevelName("WARNING"))
    logger.log(msg="train score " + str(search.score(X_train, y_train)), level=logging.getLevelName("WARNING"))
    logger.log(msg="test score " + str(search.score(X_test, y_test)), level=logging.getLevelName("WARNING"))
    logger.log(msg="test score r2 " + str(r2_score(y_test, search.predict(X_test))), level=logging.getLevelName("WARNING"))
    score = calculate_score(y_test, search.predict(X_test), y_cols)
    logger.log(msg="test score custom "+str(score), level=logging.getLevelName("WARNING"))

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


for y_cols in ([["views"], ["depth"], ["full_reads_percent"],["views", "depth", "full_reads_percent"]]):
    train_score(y_cols, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())

