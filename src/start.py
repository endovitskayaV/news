import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

from settings import LOGGING_PATH, DATA_PATH, RAW_PATH
from src.funs import str_to_list, encode_list_by_rate
from src.utils import dump, loads

logger = logging.getLogger()
formatter = logging.Formatter(
    "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
)
general_fh = logging.FileHandler(LOGGING_PATH / "logs_simple.txt")
general_fh.setFormatter(formatter)
general_fh.setLevel("INFO")
logger.addHandler(general_fh)


def prepare(df, name, is_train):
    df.sort_values('publish_date', inplace=True)
    df['Time'] = np.arange(len(df.index))


    if is_train:
        df = df[df.category.isin(
            ['5409f11ce063da9c8b588a18', '5409f11ce063da9c8b588a12', '5433e5decbb20f277b20eca9',
             '540d5ecacbb20f2524fc050a',
             '540d5eafcbb20f2524fc0509', '5409f11ce063da9c8b588a13'])]
    df = df.reset_index(drop=True)

    category_encoder = loads(DATA_PATH / "category_encoder.pickle")
    categs = category_encoder.transform(df[['category']]).toarray()
    category_feat_names = list(category_encoder.get_feature_names_out(['category']))
    category_df = pd.DataFrame(categs, columns=category_feat_names)
    df = df.merge(category_df, left_index=True, right_index=True)
    df = df.drop('category', axis=1)

    if is_train:
        # df = df[df['views'] <= 800_000]
        # df = df[df['depth'] < 1.79]
        df.loc[df['full_reads_percent'] > 100, 'full_reads_percent'] = np.nan
        df['full_reads_percent'].fillna((df['full_reads_percent'].mean()), inplace=True)

    df = encode_list_by_rate(df, 'authors', 0.03)

    df = df.apply(lambda row: str_to_list(row, 'tags'), axis=1)
    tags_encoder = loads(DATA_PATH / "tags_encoder.pickle")
    tags = tags_encoder.transform(df['tags'])
    tags_feat_names = ['tags_' + str(cls) for cls in list(tags_encoder.classes_)]
    tags_df = pd.DataFrame(tags, columns=tags_feat_names)
    df = df.merge(tags_df, left_index=True, right_index=True)
    df = df.drop('tags', axis=1)

    df['day'] = pd.to_datetime(df['publish_date']).dt.strftime("%d").astype(int)
    df['month'] = pd.to_datetime(df['publish_date']).dt.strftime("%m").astype(int)
    df['hour'] = pd.to_datetime(df['publish_date']).dt.strftime("%H").astype(int)
    df.to_csv(DATA_PATH / name, index=False)
    return df

#
# df_train = pd.read_csv(RAW_PATH / "train.csv", parse_dates=['publish_date'])
# df_train = prepare(df_train, "df_train.csv", True)
#
# df_test = pd.read_csv(RAW_PATH / "test.csv", parse_dates=['publish_date'])
# df_test = prepare(df_test, "df_test.csv", False)


df_train = pd.read_csv(DATA_PATH / "df_train.csv")
df_test = pd.read_csv(DATA_PATH / "df_test.csv")

x_cols_drop = ["views", "depth", "full_reads_percent", "publish_date", "session", "document_id", 'title', 'Time']
y_cols = ["views"]

X_train = df_train.drop(x_cols_drop, axis=1)
y_train = df_train[y_cols].values.ravel()

X_test = df_test.drop( ["publish_date", "session", "document_id", 'title', 'Time'], axis=1)
y_test = pd.read_csv(DATA_PATH / "df_test_prepared_v_real.csv")
y_test=y_test['real_views']

score_dict = {"views": 0.4, "depth": 0.3, "full_reads_percent": 0.3}

p = {'n_estimators': 500, 'max_depth': 20}
search = RandomForestRegressor(**p)
search.fit(X_train, y_train)
dump(DATA_PATH / "v.pickle", search)
logger.log(msg="train score r2 " + str(r2_score(y_train, search.predict(X_train))),
           level=logging.getLevelName("WARNING"))
logger.log(msg="test score r2 " + str(r2_score(y_test, search.predict(X_test))),
           level=logging.getLevelName("WARNING"))

col_name = 'importance'
importance_df = pd.DataFrame(search.feature_importances_, columns=[col_name],
                             index=search.feature_names_in_).sort_values(by=col_name, ascending=False)
importance_df.to_csv(DATA_PATH/ "importance.csv")

y_true = pd.read_csv(DATA_PATH / "df_test_prepared_v_real.csv")
y_true['pred_v'] = search.predict(X_test)
y_true['diff_v'] = y_true['real_views'] - y_true['pred_v']
y_true['diff_v_abs'] = (y_true['real_views'] - y_true['pred_v']).abs()
y_true.to_csv(DATA_PATH / "df_test_prepared_v_real.csv", index=False)
