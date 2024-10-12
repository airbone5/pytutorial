from flask import Flask,render_template,redirect,url_for

#app = Flask(__name__)
app = Flask(__name__, static_url_path='/static') 

@app.route('/')
def home():
  return render_template('index.html')


@app.route('/a')
def doa():
    return 'api a'

@app.route('/b')
def dob():
    #return url_for('doa') 
    return redirect(url_for('doa'))

if __name__ == '__main__':
  app.run(host='0.0.0.0')

