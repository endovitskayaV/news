import ast
import logging
import re
import urllib
from datetime import datetime, timedelta
from typing import List, Dict

import numpy as np
import pandas as pd
import spacy_stanza
import stanza
from bs4 import BeautifulSoup
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from settings import LOGGING_PATH, DATA_PATH, RAW_PATH
from src.utils import dump, loads


# funs
def str_to_list(row: Series, col_name: str) -> Series:
    row[col_name] = ast.literal_eval(row[col_name])
    return row


def encode_list_dummies(df: DataFrame, col_name: str) -> DataFrame:
    df = df.apply(lambda row: str_to_list(row, col_name), axis=1)
    categories_df = pd.get_dummies(df[col_name].apply(pd.Series).stack(), prefix=col_name).sum(level=0)
    df = df.drop(col_name, axis=1)
    df = df.join(categories_df)
    cols = [col for col in df if col.startswith(col_name)]
    df[cols] = df[cols].fillna(0)
    return df


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


def holiday_fun(row: Series, dates: Series) -> Series:
    row['is_holiday'] = row['publish_date'].date() in dates.values
    return row


def weekend_fun(row: Series) -> Series:
    row['weekday'] = row['publish_date'].date().weekday()
    return row


date_categs = {
    '10': (datetime.strptime('19-02-2000', '%d-%m-%Y').date(), datetime.strptime('19-03-2022', '%d-%m-%Y').date()),
    '20': (datetime.strptime('20-03-2022', '%d-%m-%Y').date(), datetime.strptime('08-04-2022', '%d-%m-%Y').date()),
    '30': (datetime.strptime('09-04-2022', '%d-%m-%Y').date(), datetime.strptime('19-02-2025', '%d-%m-%Y').date()),
}


def date_categ_fun(row: Series) -> Series:
    date = row['publish_date'].date()
    date_categ = -1

    for category_name, (start_date, end_date) in date_categs.items():
        if date >= start_date and date <= end_date:
            date_categ = category_name
            break

    row['date_categ'] = date_categ
    return row


def curs_fun(row: Series, dollar_df: DataFrame) -> Series:
    date = row['publish_date'].date()
    max_attempt = 5
    attempt = 0
    curs = dollar_df.loc[dollar_df['data'] == date]['curs']
    while curs.size <= 0 and attempt < max_attempt:
        date -= timedelta(days=1)
        curs = dollar_df.loc[dollar_df['data'] == date]['curs']
        attempt += 1
    curs = curs.iloc[0] if curs.size > 0 else None
    row['curs'] = curs
    return row


def len_fun(row: Series) -> Series:
    id = row['document_id'][0:24]
    content = None
    text = ""
    try:
        content = urllib.request.urlopen("https://www.rbc.ru/rbcfreenews/" + id).read()
    except Exception as e:
        logger.log(msg=e, level=logging.getLevelName("ERROR"))

    if content:
        soup = BeautifulSoup(content, 'lxml')
        text = soup.select_one('.article__text_free')
        text = text.text
    else:
        logger.log(msg="no content", level=logging.getLevelName("WARNING"))

    row['text'] = text
    row['text_length'] = len(text)
    return row


# logging
logger = logging.getLogger()
formatter = logging.Formatter(
    "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
)
general_fh = logging.FileHandler(LOGGING_PATH / "logs_test.txt")
general_fh.setFormatter(formatter)
general_fh.setLevel("INFO")
logger.addHandler(general_fh)

#transform data

df_test = pd.read_csv(RAW_PATH / "test.csv", parse_dates=['publish_date'])
df_test = df_test.apply(lambda row: len_fun(row), axis=1)

df_test.sort_values('publish_date', inplace=True)
df_test['Time'] = np.arange(len(df_test.index))

df_test = encode_dummies(df_test, 'category')
df_test = encode_list_by_rate(df_test, 'authors', 0.03)
df_test = encode_list_dummies(df_test, 'tags')
df_test['day'] = pd.to_datetime(df_test['publish_date']).dt.strftime("%d").astype(int)
df_test['month'] = pd.to_datetime(df_test['publish_date']).dt.strftime("%m").astype(int)
df_test['hour'] = pd.to_datetime(df_test['publish_date']).dt.strftime("%H").astype(int)
df_test['minute'] = pd.to_datetime(df_test['publish_date']).dt.strftime("%M").astype(int)
df_test['date'] = df_test['publish_date'].apply(lambda _date: _date.date())

holiday_dates = pd.read_csv(DATA_PATH / 'holidays.csv', sep=';')
dates = holiday_dates['date'].apply(lambda _date: datetime.strptime(_date, "%Y-%m-%d").date())
df_test = df_test.apply(lambda row: holiday_fun(row, dates), axis=1)
df_test['is_holiday'] = df_test['is_holiday'].astype(int)
df_test = df_test.apply(lambda row: weekend_fun(row), axis=1)
df_test = df_test.apply(lambda row: date_categ_fun(row), axis=1)

dollar_df = pd.ExcelFile(DATA_PATH / "dollar.xlsx")
dollar_df = dollar_df.parse("RC", parse_dates=['data'])
dollar_df['data'] = dollar_df['data'].apply(lambda _date: _date.date())
df_test = df_test.apply(lambda row: curs_fun(row, dollar_df), axis=1)
df_test['curs'].fillna((df_test['curs'].mean()), inplace=True)

x_cols_drop = ["views", "depth", "full_reads_percent", "publish_date", "session", "document_id", 'date', 'title', 'text']
y_cols = ["views", "depth", "full_reads_percent"]


# df_test = pd.read_csv(DATA_PATH / "df_test_prepared.csv", parse_dates=['publish_date'])
X = df_test.drop(x_cols_drop, axis=1)
y = df_test[y_cols]

score_dict = {"views": 0.4, "depth": 0.3, "full_reads_percent": 0.3}


search = loads(DATA_PATH / "viwes_reg.pickle")
logger.log(msg="views", level=logging.getLevelName("WARNING"))
score = calculate_score(y, search.predict(X), ['views'])
logger.log(msg="score " + str(score),level=logging.getLevelName("WARNING"))

logger.log(msg="\n", level=logging.getLevelName("WARNING"))