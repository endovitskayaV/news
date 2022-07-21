import ast
import logging
from datetime import datetime, timedelta
from typing import List, Dict

import numpy as np
import pandas as pd
# import spacy_stanza
# import stanza
import spacy_stanza
import stanza
from pandas import Series, DataFrame
from sklearn.metrics import r2_score
from textstat import textstat

from settings import LOGGING_PATH, DATA_PATH
from src.funs import str_to_list

popular_cats = ['Военная операция на Украине',
                'Политика',
                'Общество',
                'Бизнес',
                'Война санкций',
                'Пандемия коронавируса',
                'Экономика',
                'Технологии и медиа',
                'Финансы',
                'Город',
                'Высылки дипломатов',
                'Конфликт в Донбассе',
                'Протесты в Казахстане',
                'Протесты в Белоруссии',
                'Конфликт Армении и Азербайджана',
                'Рост цен на газ',
                'Выборы президента Франции',
                'Дело Навального',
                'Доходы власти',
                'Дискуссионный клуб',
                # 'Дело Порошенко и Медведчука',
                # 'Отставки губернаторов',
                # 'Протесты в Армении',
                # 'Конфликт в Афганистане',
                # 'Дело Абызова',
                # 'ПМЭФ-2022',
                # 'День выборов',
                # 'Конфликт в Нагорном Карабахе'
                ]


def replace_sub_cat(row):
    sub_cat = row['sub_cat']
    if sub_cat not in popular_cats:
        row['sub_cat'] = 'rare_sub_cat'
    return row


def fin(s):
    s = Series(data=np.arange(1, len(s.index) + 1), index=s.index)
    return s

#
# stanza.download("ru")
# nlp = spacy_stanza.load_pipeline(name="ru", lang="ru", processors="tokenize,pos,lemma")


def fun(row):
    text = row['full_text']
    i = text.find("Читайте на РБК Pro")
    row['pro_place'] = i/ len(text)
    return row


df_train = pd.read_csv(DATA_PATH / "df_text.csv", index_col=0)
df_train = df_train.apply(lambda row: fun(row), axis=1)
df_train.to_csv(DATA_PATH / "df_text_pro.csv")


# df_train = df_train[['text_length', 'avg_sentence_len', 'max_sentence_len', 'min_sentence_len', 'sub_cat', 'max_v', 'max_ctr',  'pro_div', 'related_div', 'overview_text', 'new_title']]
#
# df_train = df_train.apply(lambda row: str_to_list(row, 'new_title'), axis=1)
#
# vectorizer = TfidfVectorizer(tokenizer=identity, lowercase=False, ngram_range=(2,2), max_features=2500)
# new_title = vectorizer.fit_transform(df_train['new_title'])
# f=vectorizer.get_feature_names_out()
# dump(DATA_PATH / "vectorizer_bi.pickle", vectorizer)


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


# stanza.download("ru")
# nlp = spacy_stanza.load_pipeline(name="ru", lang="ru")


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


def ents_fun(row: Series, col_name: str) -> Series:
    text = row[col_name]
    doc = nlp(text)
    ents = [token.ent_type_ for token in doc if token.ent_type_ and token.ent_iob_ == 'B']
    row['ents'] = ents
    return row


textstat.set_lang('ru')


def readability_fun(row: Series, col_name: str) -> Series:
    text = row[col_name]
    flesch = textstat.flesch_reading_ease(text)
    ari = textstat.automated_readability_index(text)
    cli = textstat.coleman_liau_index(text)
    row['flesch'] = flesch
    row['ari'] = ari
    row['cli'] = cli
    return row


#
# def len_sent_fun(row: Series) -> Series:
#     text = row['full_text']
#     sentences = re.split(r'\. |\n+', text)
#     lens = [len(sentence.split()) for sentence in sentences if len(sentence.strip()) > 0]
#     lens = [length for length in lens if length > 10]
#     avg_len = 0 if len(lens) == 0 else sum(lens) / len(lens)
#     max_len = 0 if len(lens) == 0 else max(lens)
#     min_len = 0 if len(lens) == 0 else min(lens)
#     row['avg_sentence_len'] = avg_len
#     row['max_sentence_len'] = max_len
#     row['min_sentence_len'] = min_len
#
#     return row
#
#

#
# def div_fun(row: Series, df) -> Series:
#     id = row['document_id'][0:24]
#     related_div_len = 0
#     pro_div = 0
#     related_articles = []
#     overview_text = ''
#     try:
#         content = urllib.request.urlopen("https://www.rbc.ru/rbcfreenews/" + id).read()
#         soup = BeautifulSoup(content, 'lxml')
#         overview = soup.select_one('.article__text__overview')
#         overview_text = overview.text if overview else ''
#         text = soup.select_one('.article__text_free')
#         related_divs = text.select('.article__inline-item')
#         related_div_len = len(related_divs)
#         pro_div = len(text.select('.pro-anons'))
#
#         for related_div in related_divs:
#             a_s = related_div.select('a')
#             art = {}
#             for a in a_s:
#                 if 'article__inline-item__link' in a.attrs['class'] and 'article__inline-item__image-block' not in \
#                         a.attrs['class'] and not art.get('href', None):
#                     art['href'] = a.attrs['href']
#                     if art.get('href', None):
#                         art_id = art['href'].split('/')
#                         art_id = art_id[-1]
#                         art['raw_id'] = art_id
#                         art_id = re.sub(r'[^a-zA-Z\d]', '', art_id)
#                         art['id'] = art_id
#
#                         f = df.loc[df['document_id'].str.startswith(art_id)]
#                         art['ctr'] = f.iloc[0]['ctr'] if f.shape[0] > 0 else 0
#                         art['views'] = f.iloc[0]['views'] if f.shape[0] > 0 else 0
#                         art['cite_views'] = 0
#
#                         if art['views'] <= 0:
#                             art_id = art['href'].split('/')
#                             art_id = art_id[-1]
#                             art['raw_id'] = art_id
#                             art_id = re.sub(r'[^a-zA-Z\d]', '', art_id)
#                             art['id'] = art_id
#                             art_content = urllib.request.urlopen(art['href']).read()
#                             art_soup = BeautifulSoup(art_content, 'lxml')
#                             div = art_soup.select_one('.rbcslider__slide')
#                             url = div.attrs.get('data-shorturl', None)
#                             url = div.attrs.get('data-url', None) if not url else url
#                             if url:
#                                 paths = url.split('/')
#                                 idd = paths[-1]
#                                 content = urllib.request.urlopen("https://www.rbc.ru/redir/stat/" + idd).read()
#                                 response = json.loads(content)
#                                 art['views'] = response['show']
#                                 art['cite_views'] = 1
#
#                 elif 'article__inline-item__category' in a.attrs['class']:
#                     art['cat'] = a.text
#
#             related_articles.append(art)
#     except Exception as e:
#         logger.log(msg=e, level=logging.getLevelName("ERROR"))
#         logger.log(msg=id, level=logging.getLevelName("WARN"))
#
#     row['related_div'] = related_div_len
#     row['overview_text'] = overview_text
#     row['pro_div'] = pro_div
#     row['related_articles'] = related_articles
#     return row
#
#
# def max_v_ctr(row):
#     related_articles = row['related_articles']
#     max_v = 0
#     max_ctr = 0
#     for a in related_articles:
#         if a.get('views', 0) > max_v:
#             max_v = a['views']
#         if a.get('ctr', 0) > max_ctr:
#             max_ctr = a['ctr']
#     row['max_v'] = max_v
#     row['max_ctr'] = max_ctr
#     return row
#


def ti(row):
    title = row['title']
    new_title = []
    for t in title:
        doc = nlp(t)
        tag = doc[0].tag_
        if tag in ['VERB', "ADJ"]:
            new_title.append(t)
    row['new_title'] = new_title
    return row


logger = logging.getLogger()
formatter = logging.Formatter(
    "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
)
general_fh = logging.FileHandler(LOGGING_PATH / "logs5.txt")
general_fh.setFormatter(formatter)
general_fh.setLevel("INFO")
logger.addHandler(general_fh)

#
# import re
#
# timeline_df = pd.ExcelFile(DATA_PATH / "timeline.xlsx")
# timeline_df = timeline_df.parse("multiTimeline", parse_dates=['День'])
# timeline_df['День'] = timeline_df['День'].apply(lambda _date: _date.date())
#
# cols_to_rename={}
# for col_name in timeline_df.columns[1:]:
#     new_col_name = col_name[:col_name.find(':')].lower()
#     cols_to_rename[col_name] = new_col_name
# timeline_df  = timeline_df.rename(columns=cols_to_rename)
# timeline_columns = timeline_df.columns[1:].tolist()

df_train = pd.read_csv(DATA_PATH / "df_text.csv")
df_train = df_train.apply(lambda row: str_to_list(row, 'title'), axis=1)
df_train = df_train.apply(lambda row: ti(row), axis=1)
df_train.to_csv(DATA_PATH / "df_text.csv", index=False)

#
# df_train = pd.read_csv(RAW_PATH / "test.csv")
# #
# # dollar_df = pd.ExcelFile(DATA_PATH / "dollar.xlsx")
# # dollar_df = dollar_df.parse("RC", parse_dates=['data'])
# # dollar_df['data'] = dollar_df['data'].apply(lambda _date: _date.date())
# #
# # df_train.sort_values('publish_date', inplace=True)
# # df_train['Time'] = np.arange(len(df_train.index))
# # df_train = df_train[df_train.category.isin(
# #     ['5409f11ce063da9c8b588a18', '5409f11ce063da9c8b588a12', '5433e5decbb20f277b20eca9', '540d5ecacbb20f2524fc050a',
# #      '540d5eafcbb20f2524fc0509', '5409f11ce063da9c8b588a13'])]
# # df_train = df_train.apply(lambda row: str_to_list(row, 'title'), axis=1)
# # df_train = df_train.apply(lambda row: str_to_list(row, 'text'), axis=1)
# # df_train = df_train[df_train['views'] <= 800_000]
# # df_train = df_train[df_train['depth'] < 1.79]
# df_train.loc[df_train['full_reads_percent'] > 100, 'full_reads_percent'] = np.nan
# df_train['full_reads_percent'].fillna((df_train['full_reads_percent'].mean()), inplace=True)
# # #
# # df_train = encode_dummies(df_train, 'category')
# # df_train = encode_list_by_rate(df_train, 'authors', 0.03)
# df_train = df_train.apply(lambda row: readability_fun(row, 'full_text'), axis=1)
# df_train.to_csv(DATA_PATH / "df_text.csv", index=False)
# print('')
# # df_train['day'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%d").astype(int)
# # df_train['month'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%m").astype(int)
# # df_train['hour'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%H").astype(int)
# # df_train['minute'] = pd.to_datetime(df_train['publish_date']).dt.strftime("%M").astype(int)
# # df_train['date'] = df_train['publish_date'].apply(lambda _date: _date.date())
# #
# # holiday_dates = pd.read_csv(DATA_PATH / 'holidays.csv', sep=';')
# # dates = holiday_dates['date'].apply(lambda _date: datetime.strptime(_date, "%Y-%m-%d").date())
# # df_train = df_train.apply(lambda row: holiday_fun(row, dates), axis=1)
# # df_train['is_holiday'] = df_train['is_holiday'].astype(int)
# # df_train = df_train.apply(lambda row: weekend_fun(row), axis=1)
# # df_train = df_train.apply(lambda row: date_categ_fun(row), axis=1)
# # df_train = df_train.apply(lambda row: curs_fun(row, dollar_df), axis=1)
# # df_train['curs'].fillna((df_train['curs'].mean()), inplace=True)
#
#
# # cats = {
# #     'политика': ['5409f11ce063da9c8b588a12'],
# #     'общество': ['5433e5decbb20f277b20eca9'],
# #     'бизнес_финансы': ['540d5eafcbb20f2524fc0509', '5409f11ce063da9c8b588a18'],
# #     'экономика_медиа и технологии': ['5409f11ce063da9c8b588a13', '540d5ecacbb20f2524fc050a'],
# # }
# # cats_df_dict = {category_name: df_train[df_train['category'].isin(category_ids)] for category_name, category_ids in
# #                 cats.items()}
#
#
# def split(df):
#     df_train = DataFrame()
#     df_test = DataFrame()
#     groups = df.groupby('date')
#     for name, group in groups:
#         if group.shape[0] < 3:
#             df_train = pd.concat([group, df_train])
#         else:
#             group_train = group.sample(frac=0.8)
#             group_test = group.loc[~group.index.isin(group_train.index)]
#             df_train = pd.concat([group_train, df_train])
#             df_test = pd.concat([group_test, df_test])
#
#     return df_train, df_test
#
#
# # for category, df in cats_df_dict.items():
# # logger.log(msg="category " + category, level=logging.getLevelName("WARNING"))
# df_train = pd.read_csv(DATA_PATH / "df_train_v.csv", parse_dates=['publish_date'])
# df_test = pd.read_csv(DATA_PATH / "df_test_v.csv", parse_dates=['publish_date'])
# df_train = df_train.append(df_test)
# # df_train.to_csv(DATA_PATH / "df_text_prepared1.6.csv", index=False)
# df_train, df_test = split(df_train)
# x_cols_drop = ["views", "depth", "full_reads_percent", "publish_date", "session",
#                "document_id", 'date', 'title', 'text']
# y_cols = ["views", "depth", "full_reads_percent"]
#
# X_train = df_train.drop(x_cols_drop, axis=1)
# y_train = df_train[y_cols]
# X_test = df_test.drop(x_cols_drop, axis=1)
# y_test = df_test[y_cols]
# #


# test_title = vectorizer.transform(df_test['title'])
# test_texts_df = pd.DataFrame(test_title.toarray(), index = df_test.index)
# test_texts_df.rename(lambda col_name: "text_" + str(col_name), axis='columns', inplace=True)
# df_test = df_train.merge(test_texts_df, left_index=True, right_index=True)
#
#
#
# vectorizer = TfidfVectorizer(tokenizer=identity, lowercase=False)
# train_texts = vectorizer.fit_transform(X_train['text'])
# test_texts = vectorizer.transform(X_test['text'])
# # # dump(DATA_PATH / "text_vectorizer.pickle", vectorizer)
# #
# # vectorizer = loads(DATA_PATH / "text_vectorizer.pickle")
# # train_texts = vectorizer.transform(X_train['text'])
# # test_texts = vectorizer.transform(X_test['text'])
# #
# # train_texts_arr = train_texts.toarray()
# # train_texts_df = pd.DataFrame(train_texts_arr)
# # test_texts_df = pd.DataFrame(test_texts.toarray())
# #
# # train_texts_df.rename(lambda col_name: "text_" + str(col_name), axis='columns', inplace=True)
# # test_texts_df.rename(lambda col_name: "text_" + str(col_name), axis='columns', inplace=True)
# #
# # X_train = X_train.reset_index()
# # X_test = X_test.reset_index()
# # y_train = y_train.reset_index()
# # y_test = y_test.reset_index()
# #
# # X_train = X_train.merge(train_texts_df, left_index=True, right_index=True)
# # X_test = X_test.merge(test_texts_df, left_index=True, right_index=True)
# # X_train.drop(['title', 'text','index'], axis=1, inplace=True)
# # X_test.drop(['title',  'text', 'index'], axis=1, inplace=True)
#
# feature_array = np.array(vectorizer.get_feature_names_out())
# tfidf_sorting = np.argsort(train_texts_arr).flatten()[::-1]
# n = 500
# top_n = feature_array[tfidf_sorting][:n]
# # write_to_file(DATA_PATH / "text_top500.txt", '\n'.join(p for p in top_n))
#
# score_dict = {"views": 0.4, "depth": 0.3, "full_reads_percent": 0.3}
#
#
# # search = loads(DATA_PATH / "views_reg.pickle")
# # score = calculate_score(y_test, search.predict(X_test), ['views'])
# # print(score)
#
# def train_score(index: int, y_cols: List[str], X_train, X_test, y_train, y_test):
#     y_train = y_train[y_cols]
#     y_test = y_test[y_cols]
#
#     if y_train.shape[1] == 1:
#         y_train = y_train.values.ravel()
#
#     logger.log(msg="y_cols " + str(y_cols), level=logging.getLevelName("WARNING"))
#
#     for n_components_rate in [0]:
#         X_train_new = X_train.copy()
#         X_test_new = X_test.copy()
#         # for n_components_rate in [0.6, 0.8]:
#         #     n_components = int(n_components_rate * X_train.shape[1])
#         #     logger.log(msg="n_components " + str(n_components), level=logging.getLevelName("WARNING"))
#         #     pca = PCA(n_components=n_components)
#         #     X_train_new = pca.fit_transform(X_train)
#         #     X_test_new = pca.transform(X_test)
#         #
#         #     logger.log(msg="explained_variance_ " + str(pca.explained_variance_ratio_),
#         #                level=logging.getLevelName("WARNING"))
#         #     logger.log(msg="n_components_ " + str(pca.n_components_), level=logging.getLevelName("WARNING"))
#         #     logger.log(msg="n_features_ " + str(pca.n_features_), level=logging.getLevelName("WARNING"))
#
#         random_grid = {
#             "n_estimators": [200, 500],
#             'max_features': [0.6, 0.8, 1],
#             "max_depth": [10, 20],
#         }
#
#         # rf_random = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=random_grid,
#         #                                n_iter=100, scoring='r2',
#         #                                cv=3, verbose=2, random_state=42, n_jobs=-1,
#         #                                return_train_score=True)
#         # search = rf_random.fit(X_train_new, y_train)
#         p = {'n_estimators': 500, 'max_depth': 20}
#         search = RandomForestRegressor(**p)
#         search.fit(X_train_new, y_train)
#         dump(DATA_PATH / (str(index) + "reg4.pickle"), search)
#         # logger.log(msg="params " + str(search.best_params_), level=logging.getLevelName("WARNING"))
#         # logger.log(msg="best_score_ " + str(search.best_score_), level=logging.getLevelName("WARNING"))
#         logger.log(msg="train score r2 " + str(r2_score(y_train, search.predict(X_train_new))),
#                    level=logging.getLevelName("WARNING"))
#         logger.log(msg="train score " + str(search.score(X_train_new, y_train)),
#                    level=logging.getLevelName("WARNING"))
#         logger.log(msg="test score " + str(search.score(X_test_new, y_test)), level=logging.getLevelName("WARNING"))
#         logger.log(msg="test score r2 " + str(r2_score(y_test, search.predict(X_test_new))),
#                    level=logging.getLevelName("WARNING"))
#         score = calculate_score(y_test, search.predict(X_test_new), y_cols)
#         logger.log(msg="test score custom " + str(score), level=logging.getLevelName("WARNING"))
#         logger.log(msg="\n", level=logging.getLevelName("WARNING"))
#
#
# l1 = [["views"], ["depth"], ["full_reads_percent"], ["views", "depth", "full_reads_percent"]]
# for index, y_cols in enumerate([["views"]]):
#     train_score(index, y_cols, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
#     logger.log(msg="\n", level=logging.getLevelName("WARNING"))
#
# logger.log(msg="\n", level=logging.getLevelName("WARNING"))
