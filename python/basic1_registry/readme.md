
- [democode](./demo.py),[來自](https://stackoverflow.com/questions/15128225/python-script-to-read-and-write-a-path-to-registry)
- [參考](https://www.xingyulei.com/post/py-win-contextmenu/index.html)

```
pip install  notebook
pip freeze >requirements.txt
pip install -r requirments.txt
```
影片中的問題
1. 改完以後要重開機
1. 甚麼是(default)
```
name='' # 其實就是(預設值),也是回答影片中的問題
value=''
winreg.SetValueEx(registry_key, name, 0, winreg.REG_SZ, value)
```

