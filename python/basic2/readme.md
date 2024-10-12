## 目的

匯入外部資料,主要是CSV文件


```
pip install  notebook
pip freeze >requirements.txt
pip install -r requirments.txt
```

| Json | Dictionary |
| --- | --- |
| JSON keys can only be strings. | The dictionary’s keys can be any hashable object. |
| The keys in JSON are ordered sequentially and can be repeated. | The keys in the dictionary cannot be repeated and must be distinct. |
| The keys in JSON have a default value of undefined. | There is no default value in dictionaries. |
| The values in a JSON file are accessed by using the “.” (dot) or “\[\]” operator. | The subscript operator is used to access the values in the dictionary. For example, if ‘dict’ = ‘A’:’123R’,’B’:’678S’, we can retrieve data related by simply calling dict\[‘A’\]. |
| For the string object, we must use double quotation marks. | For string objects, we can use either a single or double quotation. |
| In JSON, the return object type is a’string’ object type. | The ‘dict’ object type is the return object type in a dictionary. |