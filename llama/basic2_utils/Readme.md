## 目的
把自己常用的OLLAMA指令,包裝在一個執行檔案(EXE)。

## 需要的套件
```
pip install pyinstaller
pyinstaller -F 單一檔案(例如ocmd.py)
pyinstaller -D 多個檔案(例如目錄)
```

執行檔案在 dist 子目錄

執行範例
```
ocmd show
```

- [ocmd_basic.py](./ocmd_basic.py): 練習影片用的程式碼 (原先是ocmd.py)
- [ocmd.py](./ocmd.py)

