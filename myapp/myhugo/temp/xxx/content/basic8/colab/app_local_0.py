#!pip install flask 
from flask import Flask
print(__name__)
app = Flask(__name__)


@app.route('/')
def home():
  return "hello"


if __name__ == '__main__':
  app.run(host='0.0.0.0')

