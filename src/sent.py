# from dostoevsky.tokenization import RegexTokenizer
# from dostoevsky.models import FastTextSocialNetworkModel
#
# from settings import DATA_PATH
#
# tokenizer = RegexTokenizer()
# tokens = tokenizer.split('всё очень плохо')  # [('всё', None), ('очень', None), ('плохо', None)]
#
# FastTextSocialNetworkModel.MODEL_PATH = str(DATA_PATH / 'fasttext-social-network-model.bin')
# model = FastTextSocialNetworkModel(tokenizer=tokenizer)
#
# messages = [
#     'Билл Гейтс заразился коронавирусом'
# ]
#
# results = model.predict(messages, k=2)
#
# for message, sentiment in zip(messages, results):
#     # привет -> {'speech': 1.0000100135803223, 'skip': 0.0020607432816177607}
#     # люблю тебя!! -> {'positive': 0.9886782765388489, 'skip': 0.005394937004894018}
#     # малолетние дебилы -> {'negative': 0.9525841474533081, 'neutral': 0.13661839067935944}]
#     print(message, '->', sentiment)
import spacy_stanza
import stanza

stanza.download("ru")
nlp = spacy_stanza.load_pipeline(name="ru", lang="ru")
doc = nlp("Билл Гейтс заразился коронавирусом от Владимира Владимировича Путина, а он от Байдена")
ents = [token.ent_iob_ + "-" + token.ent_type_ for token in doc if  token.ent_type_ and token.ent_iob_=='B']
print('')
