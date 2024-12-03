from flask import Flask,render_template,redirect,url_for

#app = Flask(__name__)
app = Flask(__name__, static_url_path='/static') # 其實第二個參數用的/static是預設值

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/js')
def xx():
  return render_template('js1.html')

@app.route('/a')
def doa():
    return 'api a'

@app.route('/b')
def dob():
    #return url_for('doa') 
    return redirect(url_for('doa'))

if __name__ == '__main__':
  app.run(host='0.0.0.0')

