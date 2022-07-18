import pandas as pd
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

from settings import DATA_PATH
from src.funs import str_to_list
# 
df_train = pd.read_csv(DATA_PATH / "df_text.csv")
df_train = df_train.apply(lambda row: str_to_list(row, 'title'), axis=1)
corpus =df_train['title']


import gensim.downloader as gensim_api
nlp = gensim_api.load("word2vec-google-news-300")
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
# word_vectors = model.wv
# model.save(str(DATA_PATH / "word2vec.model"))
# print('')

