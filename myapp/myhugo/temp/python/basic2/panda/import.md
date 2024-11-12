---
title: import
description: docker log
weight: 300
---
## 

🏷️如果不想要CELL只輸出最後一個結果,可以執行這個
```
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

## 準備工作

複製資料到目前的資料夾


```python
import os
os.popen('copy ..\\..\\dataset\\iris.csv diris.csv')
```




    <os._wrap_close at 0x232cde710a0>



## 一般讀取CSV方法
如果CSV資料沒有特殊格式,例如欄位用CSV分隔,而第一行是欄位名稱(可以沒有),那麼讀取方式比較簡單
### 讀成dict: 函數DictReader()


```python
import csv
"""
檔案abc.csv 內容
a,b,c
1,2,3
4,5,6
"""
handle=open('abc.csv')
reader1 = csv.DictReader(handle) #把每一行讀成dict

print(reader1.fieldnames) # ['a', 'b', 'c']

for row in reader1:
    print(row['a'], row['b'])
handle.close()
```

    ['a', 'b', 'c']
    1 2
    4 5
    

## 特殊CSV範例
但是有些CSV資料,並不是像上面的例子那樣安排,例如本文的測試資料[iris.csv](../../dataset/iris.csv)
```
150,4,setosa,versicolor,virginica
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
```
150代表樣本數,4代表4個欄位,後面3個代表最後一欄0,1,2的值


```python
handle=open('diris.csv')

reader1 = csv.DictReader(handle)
print(reader1)
print(reader1.fieldnames) # 雖然沒有錯誤,但是讀到的是 ['150', '4', 'setosa', 'versicolor', 'virginica']
handle.close()
```

    <csv.DictReader object at 0x00000232CDD3FF50>
    ['150', '4', 'setosa', 'versicolor', 'virginica']
    

如果要一行一行讀的話,就用函數`next()`


```python
handle=open('diris.csv')
rst=csv.reader(handle)
x=next(rst) #讀取下一行
print(x)
x=next(rst) #讀取下一行
print(x)

handle.close()
```

    ['150', '4', 'setosa', 'versicolor', 'virginica']
    ['5.1', '3.5', '1.4', '0.2', '0']
    

## 利用numpy
將內容讀成data(X),和target(y)


```python
import csv
import numpy as np
handle=open('../../dataset/iris.csv')        
data_file = csv.reader(handle)
temp = next(data_file) # 讀入第一行
n_samples = int(temp[0]) # 第一行,第1個元素是,總樣本數
n_features = int(temp[1]) # 第一行,第2個元素是,欄位個數
xx=temp[2:]
target_names = np.array(temp[2:]) # 第一行,其他欄位是輸出y號碼代表的類別
print(target_names)
data = np.empty((n_samples, n_features))
target = np.empty((n_samples,), dtype=int)
#這裡才真正讀取樣本
for i, ir in enumerate(data_file):
    data[i] = np.asarray(ir[:-1], dtype=np.float64)
    target[i] = np.asarray(ir[-1], dtype=int)

handle.close()
print("X資料前5")
print(data[0:5])
print("y資料前5")
print(target_names[target[0:5]])

```

    ['setosa' 'versicolor' 'virginica']
    X資料前5
    [[5.1 3.5 1.4 0.2]
     [4.9 3.  1.4 0.2]
     [4.7 3.2 1.3 0.2]
     [4.6 3.1 1.5 0.2]
     [5.  3.6 1.4 0.2]]
    y資料前5
    ['setosa' 'setosa' 'setosa' 'setosa' 'setosa']
    

### 利用pandas


```python
import pandas as pd
```


```python
# 這是前測
from io import StringIO
s = """1, 2
... 3, 4
... 5, 6"""
d1=pd.read_csv(StringIO(s), skiprows=[1], header=None)
print(d1)
d2=pd.read_csv(StringIO(s), skiprows=1, header=None)
print(d2)

```

           0  1
    0      1  2
    1  ... 5  6
           0  1
    0  ... 3  4
    1  ... 5  6
    


```python

df=pd.read_csv("diris.csv", skiprows=1, header=None)
# 加入欄位名稱
df.columns=['a','b','c','d','target']
#轉為numpy
data=df.iloc[:,0:3].to_numpy() 
print(data[0:2])
target=df.iloc[:,-1].to_numpy()
print(target[0:2])


```

    [[5.1 3.5 1.4]
     [4.9 3.  1.4]]
    [0 0]
    

注意 要抽取欄位a,b的時候必須是


```python
x=df[['a','b']] ## ❌不是df['a','b'] 🏷️x是一個dataframe
y=x[0:2] #看看前兩筆 y也是dataframe
print(y)
ds1=x.loc[[0,1]] #🏷️ds1==ds2
ds2=x.loc[0:1]
print(ds1)
print(ds2)
```

         a    b
    0  5.1  3.5
    1  4.9  3.0
         a    b
    0  5.1  3.5
    1  4.9  3.0
         a    b
    0  5.1  3.5
    1  4.9  3.0
    

如果想要拿到series,例如要拿到欄位a 的series型態: 那就是 


```python
pd.Series(x['a'])[0:1] # 轉成series 再利用[0:1]拿到第1 筆 
```




    0    5.1
    Name: a, dtype: float64



❓問題
df[0:2]的結果是甚麼
