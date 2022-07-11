import urllib.request
from bs4 import BeautifulSoup

contents = urllib.request.urlopen("https://www.rbc.ru/rbcfreenews/6293d0179a79477232e44877").read()
soup = BeautifulSoup(contents, 'lxml')
text = soup.select_one('.article__text_free')
len = len(text.text) if text else 0
print(contents)