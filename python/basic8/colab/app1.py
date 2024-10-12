import os
from dotenv import load_dotenv
load_dotenv()

auth_key  = os.environ.get("SECRET", None)
print(auth_key)

#!pip install flask pyngrok

from flask import Flask
from pyngrok import ngrok

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World"


ngrok.set_auth_token(auth_key)
ngrok_tunnel = ngrok.connect(5000)
print('Public URL:', ngrok_tunnel.public_url)
app.run()
 