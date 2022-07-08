import ast
import logging
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from settings import LOGGING_PATH, DATA_PATH
from src.utils import identity, dump, write_to_file, loads


# funs
def str_to_list(row: Series, col_name: str) -> Series:
    row[col_name] = ast.literal_eval(row[col_name])
    return row


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
    row['is_weekend'] = row['publish_date'].date().weekday() >= 5
    return row


# logging
logger = logging.getLogger()
formatter = logging.Formatter(
    "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
)
general_fh = logging.FileHandler(LOGGING_PATH / "logs5.txt")
general_fh.setFormatter(formatter)
general_fh.setLevel("INFO")
logger.addHandler(general_fh)

df_train = pd.read_csv(DATA_PATH / "df_text.csv", parse_dates=['publish_date'])
df_train.sort_values('publish_date', inplace=True)
df_train['Time'] = np.arange(len(df_train.index))
df_train = df_train[df_train.category.isin(
    ['5409f11ce063da9c8b588a18', '5409f11ce063da9c8b588a12', '5433e5decbb20f277b20eca9', '540d5ecacbb20f2524fc050a',
     '540d5eafcbb20f2524fc0509', '5409f11ce063da9c8b588a13'])]
df_train = df_train.apply(lambda row: str_to_list(row, 'title'), axis=1)
df_train = df_train[df_train['views'] < 800_000]
df_train = df_train[df_train['depth'] < 1.79]
df_train.loc[df_train['full_reads_percent'] > 100, 'full_reads_percent'] = np.nan
df_train['full_reads_percent'].fillna((df_train['full_reads_percent'].mean()), inplace=True)

# df_train = encode_dummies(df_train, 'category')
df_train = encode_list_by_rate(df_train, 'authors', 0.03)
df_train = encode_list_by_rate(df_train, 'tags', 0.01)
df_train['day'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%d").astype(int)
df_train['month'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%m").astype(int)
df_train['hour'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%H").astype(int)
df_train['minute'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%M").astype(int)
df_train['date'] = df_train['publish_date'].apply(lambda _date: _date.date())

holiday_dates = pd.read_csv(DATA_PATH / 'holidays.csv', sep=';')
dates = holiday_dates['date'].apply(lambda _date: datetime.strptime(_date, "%Y-%m-%d").date())
df_train = df_train.apply(lambda row: holiday_fun(row, dates), axis=1)
df_train['is_holiday'] = df_train['is_holiday'].astype(int)

df_train = df_train.apply(lambda row: weekend_fun(row), axis=1)
df_train['is_weekend'] = df_train['is_weekend'].astype(int)

# categs = {
#     'политика': ['5409f11ce063da9c8b588a12'],
#     'общество': ['5433e5decbb20f277b20eca9'],
#     'бизнес_финансы': ['540d5eafcbb20f2524fc0509', '5409f11ce063da9c8b588a18'],
#     'экономика': ['5409f11ce063da9c8b588a13', ],
#     'медиа и технологии': ['540d5ecacbb20f2524fc050a'],
# }
# df_dict = {category_name: df_train[df_train['category'].isin(category_ids)] for category_name, category_ids in
#            categs.items()}


def split(df):
    df_train = DataFrame()
    df_test = DataFrame()
    groups = df.groupby('date')
    for name, group in groups:
        if group.shape[0] < 3:
            df_train = pd.concat([group, df_train])
        else:
            group_train = group.sample(frac=0.8)
            group_test = group.loc[~group.index.isin(group_train.index)]
            df_train = pd.concat([group_train, df_train])
            df_test = pd.concat([group_test, df_test])

    return df_train, df_test


# logger.log(msg="category " + category, level=logging.getLevelName("WARNING"))

df_train, df_test = split(df_train)
x_cols_drop = ['is_weekend', 'views', 'category', 'depth', "full_reads_percent", "publish_date", "session",
               "document_id", 'date']
y_cols = ["views", "depth", "full_reads_percent"]

X_train = df_train.drop(x_cols_drop, axis=1)
y_train = df_train[y_cols]
X_test = df_test.drop(x_cols_drop, axis=1)
y_test = df_test[y_cols]

vectorizer = TfidfVectorizer(tokenizer=identity, lowercase=False)
# vectorizer = loads(DATA_PATH / "views_vectorizer.pickle")
train_texts = vectorizer.fit_transform(X_train['title'])
test_texts = vectorizer.transform(X_test['title'])
dump(DATA_PATH / ("category" + "vectorizer.pickle"), vectorizer)

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
X_train.drop(['title', 'index'], axis=1, inplace=True)
X_test.drop(['title', 'index'], axis=1, inplace=True)

feature_array = np.array(vectorizer.get_feature_names_out())
tfidf_sorting = np.argsort(train_texts_arr).flatten()[::-1]
n = 500
top_n = feature_array[tfidf_sorting][:n]
write_to_file(DATA_PATH / "top500.txt", '\n'.join(p for p in top_n))

score_dict = {"views": 0.4, "depth": 0.3, "full_reads_percent": 0.3}


# search = loads(DATA_PATH / "views_reg.pickle")
# score = calculate_score(y_test, search.predict(X_test), ['views'])
# print(score)

def train_score(index: int, y_cols: List[str], X_train, X_test, y_train, y_test):
    y_train = y_train[y_cols]
    y_test = y_test[y_cols]

    if y_train.shape[1] == 1:
        y_train = y_train.values.ravel()

    logger.log(msg="y_cols " + str(y_cols), level=logging.getLevelName("WARNING"))

    for n_components_rate in [0]:
        X_train_new = X_train.copy()
        X_test_new = X_test.copy()
        # for n_components_rate in [0.6, 0.8]:
        #     n_components = int(n_components_rate * X_train.shape[1])
        #     logger.log(msg="n_components " + str(n_components), level=logging.getLevelName("WARNING"))
        #     pca = PCA(n_components=n_components)
        #     X_train_new = pca.fit_transform(X_train)
        #     X_test_new = pca.transform(X_test)
        #
        #     logger.log(msg="explained_variance_ " + str(pca.explained_variance_ratio_),
        #                level=logging.getLevelName("WARNING"))
        #     logger.log(msg="n_components_ " + str(pca.n_components_), level=logging.getLevelName("WARNING"))
        #     logger.log(msg="n_features_ " + str(pca.n_features_), level=logging.getLevelName("WARNING"))

        random_grid = {
            "n_estimators": [200, 500],
            'max_features': [0.6, 0.8, 1],
            "max_depth": [10, 20],
        }

        # rf_random = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=random_grid,
        #                                n_iter=100, scoring='r2',
        #                                cv=3, verbose=2, random_state=42, n_jobs=-1,
        #                                return_train_score=True)
        # search = rf_random.fit(X_train_new, y_train)
        p = {'n_estimators': 500, 'max_features': 0.8, 'max_depth': 20}
        search = RandomForestRegressor(**p)
        search.fit(X_train_new, y_train)
        dump(DATA_PATH / (str(index) + "reg.pickle"), search)
        # logger.log(msg="params " + str(search.best_params_), level=logging.getLevelName("WARNING"))
        # logger.log(msg="best_score_ " + str(search.best_score_), level=logging.getLevelName("WARNING"))
        logger.log(msg="train score r2 " + str(r2_score(y_train, search.predict(X_train_new))),
                   level=logging.getLevelName("WARNING"))
        logger.log(msg="train score " + str(search.score(X_train_new, y_train)),
                   level=logging.getLevelName("WARNING"))
        logger.log(msg="test score " + str(search.score(X_test_new, y_test)), level=logging.getLevelName("WARNING"))
        logger.log(msg="test score r2 " + str(r2_score(y_test, search.predict(X_test_new))),
                   level=logging.getLevelName("WARNING"))
        score = calculate_score(y_test, search.predict(X_test_new), y_cols)
        logger.log(msg="test score custom " + str(score), level=logging.getLevelName("WARNING"))
        logger.log(msg="\n", level=logging.getLevelName("WARNING"))


l1 = [["views"], ["depth"], ["full_reads_percent"], ["views", "depth", "full_reads_percent"]]
for index, y_cols in enumerate([["views"]]):
    train_score(index, y_cols, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
    logger.log(msg="\n", level=logging.getLevelName("WARNING"))

logger.log(msg="\n", level=logging.getLevelName("WARNING"))
