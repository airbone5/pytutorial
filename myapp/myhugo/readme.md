
## 背景
jupyter 轉換為其他格式的指令
```
jupyter nbconvert <path-to-notebook> --to html --template reveal
```
[nbconvert](https://www.vincent-lunot.com/post/toward-publishing-jupyter-notebooks-with-hugo/)

venv環境中 的pyinstaller 打包 tohugo.py  
requirements.txt
```
nb2hugo
argparse
```
會找不到templates 需要:
1. 整個`C:\Users\linchao\AppData\Local\Programs\Python\Python312\share\jupyter\nbconvert` 複製到
`C:\Users\linchao\AppData\Roaming\jupyter`

## myhugo

### 目錄結構
```
D:.
├─data (打包用的)
├─temp (測試的子目錄)
└─舊版 
```

需要兩個檔案,一個子目錄data
- subhugo.py
- myhugo.py
- data\
### 打包
必須先執行venv (同時裡面要有pyinstaller)否則執行結果很慢
```
pyinstaller --add-data=data;data -F myhugo.py
# 意思是把上面的data 應設定exe裡面的data目錄
```
demo:
```
python myhugo.py newsite temp\xxx
python myhugo.py tohugo --srcdir d:\work\python --destdir xxx

python myhugo.py fixcontent --srcdir temp\python
```
exe demo
```
dist\myhugo newsite temp\xxx 
dist\myhugo tohugo --srcdir d:\work\python --destdir temp\xxx
```