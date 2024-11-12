python 常用指令
```
pip freeze > requirements.txt

# 安裝
pip install -r requirements.txt

# 移除
pip uninstall -r requirements.txt

# 一次全部移除
pip uninstall -r requirements.txt -y
```


## notebook

```
pip install  notebook
pip freeze >requirements.txt
pip install -r requirments.txt
```



## git 
```
git add .
git commit -m "訊息"
git push
git rm --cached -r .
git reset
```


| Json | Dictionary |
| --- | --- |
| JSON keys can only be strings. | The dictionary’s keys can be any hashable object. |
| The keys in JSON are ordered sequentially and can be repeated. | The keys in the dictionary cannot be repeated and must be distinct. |
| The keys in JSON have a default value of undefined. | There is no default value in dictionaries. |
| The values in a JSON file are accessed by using the “.” (dot) or “\[\]” operator. | The subscript operator is used to access the values in the dictionary. For example, if ‘dict’ = ‘A’:’123R’,’B’:’678S’, we can retrieve data related by simply calling dict\[‘A’\]. |
| For the string object, we must use double quotation marks. | For string objects, we can use either a single or double quotation. |
| In JSON, the return object type is a’string’ object type. | The ‘dict’ object type is the return object type in a dictionary. |