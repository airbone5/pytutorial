
from flask import Flask
from pyngrok import ngrok
from google.colab import userdata
auth_key=userdata.get('mykey')

app = Flask(__name__)

@app.route("/")
def hello():
    return "你好"


ngrok.set_auth_token(auth_key)
ngrok_tunnel = ngrok.connect(5000)
print('Public URL:', ngrok_tunnel.public_url)
app.run()