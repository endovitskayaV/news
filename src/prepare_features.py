import logging
import re
import urllib

import pandas as pd
import spacy_stanza
import stanza
from bs4 import BeautifulSoup
from pandas import Series

from settings import RAW_PATH, DATA_PATH, LOGGING_PATH

logger = logging.getLogger()
formatter = logging.Formatter(
    "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
)
general_fh = logging.FileHandler(LOGGING_PATH / "logs.txt")
general_fh.setFormatter(formatter)
general_fh.setLevel("INFO")
logger.addHandler(general_fh)


def add_features(row: Series) -> Series:
    id = row['document_id'][0:24]
    text = ""
    sub_cat = ""
    text_length = 0
    a = 0
    p = 0
    img = 0
    video = 0
    avg_sentence_len = 0
    max_sentence_len = 0
    min_sentence_len = 0

    try:
        content = urllib.request.urlopen("https://www.rbc.ru/rbcfreenews/" + id).read()
        soup = BeautifulSoup(content, 'lxml')
        article_text = soup.select_one('.article__text_free')
        text = article_text.text
        text_length = len(text)
        sub_cat = soup.select_one('.article__header__category')
        sub_cat = sub_cat.text
        img = int(len(article_text.select('.article__main-image__image')) > 0)
        video = int(len(article_text.select('.article__inline-video')) > 0)
        p = len(article_text.select('p'))
        a = article_text.select('a')
        a = len([link for link in a if not link.attrs.get('class', None)])

        sentences = re.split(r'\. |\n+', text)
        lens = [len(sentence.split()) for sentence in sentences if len(sentence.strip()) > 0]
        lens = [length for length in lens if length > 10]
        avg_sentence_len = 0 if len(lens) == 0 else sum(lens) / len(lens)
        max_sentence_len = 0 if len(lens) == 0 else max(lens)
        min_sentence_len = 0 if len(lens) == 0 else min(lens)

    except Exception as e:
        logger.log(msg=e, level=logging.getLevelName("ERROR"))
        logger.log(msg=id, level=logging.getLevelName("WARN"))

    row['text'] = text
    row['text_length'] = text_length
    row['a'] = a
    row['p'] = p
    row['img'] = img
    row['video'] = video
    row['sub_cat'] = sub_cat
    row['avg_sentence_len'] = avg_sentence_len
    row['max_sentence_len'] = max_sentence_len
    row['min_sentence_len'] = min_sentence_len
    return row


stanza.download("ru")
nlp = spacy_stanza.load_pipeline(name="ru", lang="ru")


def add_ents(row: Series) -> Series:
    text = row['title']
    doc = nlp(text)
    ents = [token.ent_type_ for token in doc if token.ent_type_ and token.ent_iob_ == 'B']
    row['ents'] = ents
    return row


def add_pro_features(row: Series) -> Series:
    text = row['text']

    pro_index = -1
    full_cleaned_lines = []
    cleaned_lines_before_pro = []
    for index, line in enumerate(text.split('\n')):
        if line.startswith('Читайте на РБК Pro') or line.startswith('Россия Москва Мир'):
            pro_index = index
        if line.startswith(' ') or \
                line.startswith('Читайте на РБК Pro') or \
                line.startswith('www.adv.rbc.ru') or \
                line.startswith('Россия Москва Мир') or \
                line.startswith('0 (за сутки)') or \
                line.startswith('Читать подробнее') or \
                line.startswith('Источник:') or \
                line.startswith('Pro') or \
                line.startswith('www') or \
                len(line.strip(" ").split(" ")) == 1:
            pass
        else:
            full_cleaned_lines.append(line)
            if pro_index == -1:
                cleaned_lines_before_pro.append(line)

    full_cleaned_text = " ".join(full_cleaned_lines)
    cleaned_text_before_pro = " ".join(cleaned_lines_before_pro)

    row['full_cleaned_text'] = full_cleaned_text
    row['before_pro_text_ratio'] = len(cleaned_text_before_pro) / max(1, len(full_cleaned_text))
    row['cleaned_text_length'] = len(full_cleaned_text)
    row['cleaned_text_before_pro_length'] = len(cleaned_text_before_pro)

    return row


# CHANGE to test.csv or validation.csv
input_df_name = "train2.csv"
output_df_name = "enriched_" + input_df_name

df = pd.read_csv(RAW_PATH / input_df_name)
df = df.apply(lambda row: add_features(row), axis=1)
df = df.apply(lambda row: add_ents(row), axis=1)
df = df.apply(lambda row: add_pro_features(row), axis=1)
df.to_csv(DATA_PATH / output_df_name)
