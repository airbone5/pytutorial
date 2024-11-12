
from flask import Flask,render_template
from pyngrok import ngrok
from google.colab import userdata
auth_key=userdata.get('mykey')

app = Flask(__name__)

@app.route('/')
def home():
  return render_template('index.html')


ngrok.set_auth_token(auth_key)
ngrok_tunnel = ngrok.connect(5000)
print('Public URL:', ngrok_tunnel.public_url)
app.run()