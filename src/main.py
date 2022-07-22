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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from spacy.lang.ru.stop_words import STOP_WORDS

from settings import RAW_PATH, LOGGING_PATH, DATA_PATH
from src.utils import identity, dump, write_to_file, loads

logger = logging.getLogger()
formatter = logging.Formatter(
    "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
)
general_fh = logging.FileHandler(LOGGING_PATH / "logs.txt")
general_fh.setFormatter(formatter)
general_fh.setLevel("INFO")
logger.addHandler(general_fh)

# df_train = pd.read_csv(RAW_PATH / "train.csv", index_col=0)
df_train = pd.read_csv(DATA_PATH / "df_text.csv")
# df_train.reset_index(inplace=True)

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


# df_train = encode_dummies(df_train, 'category')
# df_train = encode_list_by_rate(df_train, 'authors', 0.03)
# df_train['day'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%d").astype(int)
# df_train['month'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%m").astype(int)

# title_vectorized = loads(DATA_PATH / "title_vectorized.pickle")
#df_title_vectorized = pd.DataFrame.sparse.from_spmatrix(title_vectorized).add_prefix('title_')


#df_train=df_train.merge(df_title_vectorized, left_index=True, right_index=True)

# nltk.download('stopwords')
# STOP_WORDS = set(stopwords.words("russian"))

stanza.download("ru")
nlp = spacy_stanza.load_pipeline(name="ru", lang="ru", processors="tokenize,pos,lemma")


def clean_text(row: Series, col_name: str) -> Series:
    text = row[col_name]
    text = text.lower()
    text = re.sub(r'[^A-zА-я\s]+', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    row['title_cleaned'] = normalize_text(text)
    return row

d[0].pos_='VERB'
d[0].tag_='ADJ'
def normalize_text(text: str) -> List[str]:
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc]
    return [word for word in lemmatized_words if word not in STOP_WORDS]


df_train = df_train.apply(lambda row: clean_text(row, 'title'), axis=1)
df_train.to_csv(DATA_PATH/"df_text.csv")

def str_to_list(row: Series, col_name: str) -> Series:
      row[col_name] = ast.literal_eval(row[col_name])
      return row

# #
# df_train = pd.read_csv(DATA_PATH/"df_text.csv")
# df_train = df_train.apply(lambda row: str_to_list(row, 'text'), axis=1)

# vectorizer = TfidfVectorizer(tokenizer=identity, lowercase=False, min_df=0.007, max_df=0.9)
# tra = vectorizer.fit_transform(df_train['title'])
# dump(DATA_PATH / "views_vectorizer.pickle", vectorizer)
#
# feature_array = np.array(vectorizer.get_feature_names())
# tfidf_sorting = np.argsort(tra.toarray()).flatten()[::-1]
# n = 500
# top_n = feature_array[tfidf_sorting][:n]
# write_to_file(DATA_PATH / "top.txt", '\n'.join(p for p in top_n))

X=df_train
#df_train.drop([col for col in df_train.columns if not col.startswith("title_")],axis=1)
#X = df_train.drop(["views", "depth", "full_reads_percent", "title", "publish_date", "session", "tags", "document_id"], axis=1)
y = df_train[["views", "depth", "full_reads_percent"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

vectorizer = loads(DATA_PATH / "views_vectorizer.pickle")
X_train = vectorizer.transform(X_train['title'])
X_test= vectorizer.transform(X_test['title'])


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

    estimator = RandomForestRegressor()

    param_grid = {
        "n_estimators": [100, 200, 500, 1000],
        "max_features": [1.0, "sqrt", "log2"],
        "min_samples_split": [10, 100, 200, 500, 1000],
        "bootstrap": [True, False],
        "max_depth": [2,5,10,100,200],
    }
    grid = GridSearchCV(estimator, param_grid, n_jobs=-1, cv=5)
    grid.fit(X_train, y_train)
    logger.log(msg="y_cols "+str(y_cols), level=logging.getLevelName("WARNING"))
    logger.log(msg="grid.best_score_ "+str(grid.best_score_), level=logging.getLevelName("WARNING"))
    logger.log(msg="grid.best_params_ "+str(grid.best_params_), level=logging.getLevelName("WARNING"))

    regr = RandomForestRegressor(**grid.best_params_)
    regr.fit(X_train, y_train)

    pred = regr.predict(X_test)
    logger.log(msg="train score " + str(regr.score(X_train, y_train)), level=logging.getLevelName("WARNING"))
    logger.log(msg="train score r2 " + str(r2_score(y_train, regr.predict(X_train))), level=logging.getLevelName("WARNING"))
    logger.log(msg="test score r2 " + str(r2_score(y_test, pred)), level=logging.getLevelName("WARNING"))
    score = calculate_score(y_test, pred, y_cols)
    logger.log(msg="test score "+str(score), level=logging.getLevelName("WARNING"))

    # col_name = 'importance'
    # importance_df = pd.DataFrame(regr.feature_importances_, columns=[col_name],
    #                              index=regr.feature_names_in_).sort_values(by=col_name, ascending=False)
    # importance_df.to_csv(DATA_PATH/ (str(round(time.time() * 1000))+"old_importance.csv"))


for y_cols in ([["views"], ["depth" ], ["full_reads_percent"],["views", "depth", "full_reads_percent"]]):
    train_score(y_cols, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())

