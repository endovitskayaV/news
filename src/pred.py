import pandas as pd

from settings import DATA_PATH
from src.utils import loads

X_test= pd.read_csv(DATA_PATH / "test.csv", sep=',')



search = loads(DATA_PATH / "search.pickle")
y_test=search.pred(X_test)
X_pred = X_test.merge(y_test, left_index=True, right_index=True)
X_pred.to_csv(DATA_PATH/"pred.csv", index=False, sep=',')
X_pred.to_csv(DATA_PATH/"pred2.csv", index=True, sep=',')