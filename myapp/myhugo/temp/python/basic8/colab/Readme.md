---
title: Readme
description: docker log
weight: 300
---
- [setup](https://dashboard.ngrok.com/get-started/setup/windows)
- [參考](https://dashboard.ngrok.com/get-started/your-authtoken)
- 主要程式碼[app1.py](./app1.py) 需要將auth token 填入.env 更改`envdemo`為`.env`

```
#!pip install flask pyngrok

from flask import Flask
from pyngrok import ngrok

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World"


ngrok.set_auth_token("自己的授權碼")
ngrok_tunnel = ngrok.connect(5000)
print('Public URL:', ngrok_tunnel.public_url)
app.run()
```  