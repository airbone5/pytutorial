from flask import Flask
from pyngrok import ngrok

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World"

if __name__ == '__main__':
  ngrok.set_auth_token("你的授權碼")
  ngrok_tunnel = ngrok.connect(5000)
  print('Public URL:', ngrok_tunnel.public_url)
  app.run()
