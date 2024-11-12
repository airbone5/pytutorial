---
title: readme
description: docker log
weight: 300
---
```cmd
py -m venv a1
.\a1\Scripts\activate.bat
pip3 show numpy
```
出現(not found) 沒有安裝 
```
pip3 install numpy

```
離開環境
```
deactivate
pip3 show numpy
```
仍然顯示沒有安裝,因為nump3安裝在testenv的環境下。

刪除虛擬環境
1. 脫離環境(deactivate)
1. 刪除專案下的資料夾testenv

## 管理套件
```
pip freeze > requirements.txt
pip install -r requirements.txt
```