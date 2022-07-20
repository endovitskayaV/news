import ast
import json
import logging
import urllib
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
from bs4 import BeautifulSoup
from pandas import Series, DataFrame
from sklearn.metrics import r2_score


# funs
def str_to_list(row: Series, col_name: str) -> Series:
    row[col_name] = ast.literal_eval(row[col_name])
    return row


def str_to_json(row: Series, col_name: str) -> Series:
    value = row[col_name]
    value = value.replace("'", "\"")
    row[col_name] = json.loads(value)
    return row


def words_amount(row: Series, text_col_name: str, len_col_name: str) -> Series:
    words = row[text_col_name].split('\n')[0].split()
    length = len(words)
    row[len_col_name] = length
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
    try:
        content = urllib.request.urlopen("https://www.rbc.ru/rbcfreenews/" + id).read()
    except Exception as e:
        logger.log(msg=e, level=logging.getLevelName("ERROR"))

    if content:
        soup = BeautifulSoup(content, 'lxml')
        text = soup.select_one('.article__text_free')
        text = text.text
    else:
        text = ""

    row['text'] = text
    return row


def timeline_fun(row: Series, timeline_df: DataFrame, timeline_columns: List[str]) -> Series:
    date = row['publish_date'].date()
    title = row['title']

    # dict_replace = {"путина": "путин",
    #                 "ес": "евросоюз",
    #                 "лавр": "лавров",
    #                 "ск": "комитет",
    #                 "германие": "германия",
    #                 "британие": "британия",
    #                 "медвед": "медведев",
    #                 "крыма": "крым",
    #                 "украиная": "украина",
    #                 "кадыр": "кадыров",
    #                 "кремло": "кремль",
    #                 "цб": "центробанк",
    #                 "киева": "киев",
    #                 "великобритание": "британия",
    #                 "кремля": "кремль",
    #                 "санкция": "санкции",
    #                 "шойг": "шойгу",
    #                 "володина": "володин",
    #                 "шольцо": "шольц",
    #                 "мариупол": "мариуполь",
    #                 "мишустина": "мишустин",
    #                 "мидлион": "миллион",
    #                 "осетие": "осетия",
    #                 "бастрыкина": "бастрыкин",
    #                 "израиля": "израиль",
    #                 "силуан": "силуанов",
    #                 "беженец": "беженцы",
    #                 "фейка": "фейк",
    #                 "кулеб": "кулеба",
    #                 "румыние": "румыния",
    #                 "лдпра": "лдпр",
    #                 "псак": "псаки",
    #                 "словакий": "словакия",
    #                 "армение": "армения",
    #                 "ереванин": "ереван",
    #                 "фаса": "фас",
    #                 "стамбуля": "стамбул",
    #                 "австралие": "австралия",
    #                 "кита": "кит",
    #                 "последствие": "последствия",
    #                 "херсонский": "херсон",
    #                 "кадыровый": "кадыров"}
    #
    # for word, repl in dict_replace.items():
    #     if word in title:
    #         title = [w if w != word else repl for w in title]
    # row['title'] = title

    max_popularity = 0
    popularity_words = []
    # sum_popularity = 0
    for timeline_column in timeline_columns:
        if timeline_column in title:
            popularity_words.append(timeline_column)
            popularity = timeline_df.loc[timeline_df['День'] == date][timeline_column]
            popularity = popularity.iloc[0] if popularity.size > 0 else 0
            if popularity > max_popularity:
                max_popularity = popularity

    row['popularity'] = max_popularity
    row['popularity_words'] = popularity_words
    return row
