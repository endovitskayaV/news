# -*- coding: cp1251 -*-
from textstat import textstat

test_data = """


В Берлине прошли переговоры глав Франции, Германии и Польши, где они обсуждали украинский кризис и стратегию НАТО. Страны должны корректировать стратегию НАТО из-за новых вызовов. Что сказали Макрон, Шольц и Дуда — в видео РБК

 







                    Франция, Польша и ФРГ призвали НАТО изменить стратегию сдерживания
                                    


Политика 







"""
textstat.set_lang('ru')
print(textstat.automated_readability_index(test_data))
print(textstat.coleman_liau_index(test_data))
