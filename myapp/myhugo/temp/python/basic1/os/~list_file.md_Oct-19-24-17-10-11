---
title: list_file
description: docker log
weight: 300
---
List Files in a Directory Using Os Module in Python
We can use these 3 methods of the OS module, to get a list of files in a directory.

- os.listdir() Method
- os.walk() Method
- os.scandir() Method


```python
import os
```


```python

## 不指定參數就是current dir
os.listdir()
## 指定參數
path = "C://Users//linchao//Desktop"
os.listdir(path)

```




    ['desktop.ini', 'Docker Desktop.lnk']




```python
# 結果看起來和上面一樣,但是在py中,上面程式碼不會出現結果,這裡利用print
# 輸出
dir_list=os.listdir(path)

print("Files and directories in '", path, "' :")
print(dir_list)
```

    Files and directories in ' C://Users//linchao//Desktop ' :
    ['desktop.ini', 'Docker Desktop.lnk']
    


```python
os.getcwd()
```




    'd:\\work\\python\\basic1_os'




```python
os.path.abspath(os.getcwd())
```




    'd:\\work\\python\\basic1_os'




```python
os.path.relpath(os.getcwd()) # 結果出現一點,代表getcwd()拿到的當前目錄,就是 `工作子目錄`.
```




    '.'




```python
tuple(os.walk('.'))
```




    (('.',
      [],
      ['closeenv.bat',
       'demo_read_1.py',
       'demo_read_2.py',
       'list_file.ipynb',
       'readme.md',
       'requirements.txt',
       'runenv.bat']),)



```
(('.', 🏷️根目錄
  [], 🏷️子目錄
  ['closeenv.bat', 🏷️檔案
   'demo_read_1.py',
   'demo_read_2.py',
   'list_file.ipynb',
   'readme.md',
   'requirements.txt',
   'runenv.bat']),)
```   


```python
#import os
path='.'
for folder, subfolders, filenames in os.walk(path, topdown=True):
    for subfolder in subfolders:
        for filename in filenames:
            list=list +[os.path.join(folder,subfolder,filename)]
            #print(f'{folder}/{subfolder}內含檔案為：{filename}')
print(list)           
```

    .\closeenv.bat
    .\demo_read_1.py
    .\demo_read_2.py
    .\list_file.ipynb
    .\readme.md
    .\requirements.txt
    .\runenv.bat
    

## 內建的路徑物件


```python
from pathlib import Path
p = Path('/home/window')
#p WindowsPath('/home/window')
p=p / 'xx'  #注意看上一行,P不是字串,而是一個WindowsPath物件 這裡的除號會被p這個物件解讀為增加路徑
str(p)
```




    '\\home\\window\\xx'




```python
# 找到相對路徑
p.relative_to(Path('/home'))

```




    WindowsPath('window/xx')


