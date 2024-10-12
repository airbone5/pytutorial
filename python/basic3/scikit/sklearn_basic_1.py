
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
iris = load_iris()

X,y=iris.data,iris.target # X,y 分別是輸入X和輸出y
print(X)
