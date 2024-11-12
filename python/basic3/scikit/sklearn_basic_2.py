from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import mean_absolute_error
iris = load_iris()
pipe = Pipeline(steps=[
   ('select', SelectKBest(k=2)),
   ('clf', LogisticRegression())])
pipe.fit(iris.data, iris.target)
pipe[:-1].get_feature_names_out() # 最後一個步驟中的零件,調用主成分名稱
yhat=pipe.predict(iris.data)
print(yhat-iris.target)
MAE=mean_absolute_error(yhat,iris.target)
print(MAE)
