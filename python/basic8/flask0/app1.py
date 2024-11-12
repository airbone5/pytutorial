#from flask import Flask, render_template, jsonify, request
from flask import Flask
app = Flask(__name__)


@app.route('/')
def home():
  return "hello"

@app.route('/name')
def name():
  return "john"


if __name__ == '__main__':
  app.run(host='0.0.0.0',debug=True)
  #app.run(host='0.0.0.0', port='8080', debug=True)
