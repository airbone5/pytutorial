---
title: Pyinstall Primer
description: Usage_Pyinstall
tags: []
categories: []
series: []
editext: md
---
<!--more-->

## 加入txt檔案的方法:
```sh
pyinstaller --add-data=data;. -F newsite.py 
如果要debug
pyinstaller --windowed --add-data=data;. -F newsite.py 
```

```
.
├── data 
│      │-- file.txt  
├── test.py
```
test.py 內容
```
import os
import sys
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.environ.get("_MEIPASS2",os.path.abspath("."))

    return os.path.join(base_path, relative_path)

# def resource_path(relative_path):
#     """ Get absolute path to resource, works for dev and for PyInstaller """
#     try:
#         # PyInstaller creates a temp folder and stores path in _MEIPASS
#         base_path = sys._MEIPASS
#     except Exception:
#         base_path = os.path.abspath(".")  
#     return base_path

#h=open('./file.txt','r')


h=open(resource_path('file.txt'),'r')
txt=h.read()
print(txt)
h.close()

```

