---
title: readme
description: docker log
weight: 300
---
## [scikit-learn 參考](https://scikit-learn.org/1.5/install.html)

主要的小型標準數據集包括：

- Iris（鳶尾花）數據集
- Digits（手寫數字）數據集
- Boston Housing（波士頓房價）數據集 house_prices
- Breast Cancer Wisconsin（乳腺癌）數據集
- Wine（葡萄酒）數據集
- Linnerud 數據集
- Diabetes（糖尿病）數據集

example
```python
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_boston_house_prices
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_linnerud_exercise
from sklearn.datasets import load_linnerud_pyhsiological
from sklearn.datasets import load_wine_data
```

```
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
iris = load_iris()
pipe = Pipeline(steps=[
   ('select', SelectKBest(k=2)),
   ('clf', LogisticRegression())])
pipe.fit(iris.data, iris.target)
Pipeline(steps=[('select', SelectKBest(...)), ('clf', LogisticRegression(...))])
pipe[:-1].get_feature_names_out()
array(['x2', 'x3'], ...)
```