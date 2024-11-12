## 範例程式碼
- [本地找檔案](./demo_findfile.py)
- [分享檔案](./demo_share.py)

## 測試檔案
- [test1 產生OCR](./test1.py)
- [test2 讀入名冊](./test2.py)
- [test3 json response](./test3.py)
- [test4 產生json array](./test4.py)
- [test5 response with json ](./test5.py) 但是是中文亂碼
- [test7 參數測試](./test7.py)
- [test8 測試把OCR放到網頁,redirect_url](./test8.py)
- [test9 把測試8的兩個API變成一個](./test9.py)


## 主程式
main.py

## more
[一個系列中的一篇url_for](https://hackmd.io/@shaoeChen/BkApyHhgf?type=view)
[參考](https://realpython.com/python-code-image-generator/)

```
1

Navigate the folder where images are located. Then start a simple file server with SimpleHTTPServer (2), or http.server (3).

$ cd images_directory
$ python3 -m http.server  // python2 -m SimpleHTTPServer
this will let you, access images via web with urls like, 'http://localhost:8000/image.jpg'

EDIT:

from mimetypes import MimeTypes
mime = MimeTypes()

#打包URL資源
def make_url(mime_type, bin_data):
    return 'data:'+f_mime+';base64, '+bin_data
# 自動猜測型態
your_files_mimetype = mime.guess_type(path_to_your_file)[0] #3 returns a tuple

with open(path_to_your_file, 'rb') as f:
      data = f.read().encode('base64')
      url = make_url(your_files_mimetype, data)
```      