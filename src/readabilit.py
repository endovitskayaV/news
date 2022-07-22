# -*- coding: cp1251 -*-
from textstat import textstat

test_data = """


� ������� ������ ���������� ���� �������, �������� � ������, ��� ��� ��������� ���������� ������ � ��������� ����. ������ ������ �������������� ��������� ���� ��-�� ����� �������. ��� ������� ������, ����� � ���ࠗ � ����� ���

 







                    �������, ������ � ��� �������� ���� �������� ��������� �����������
                                    


�������� 







"""
textstat.set_lang('ru')
print(textstat.automated_readability_index(test_data))
print(textstat.coleman_liau_index(test_data))
