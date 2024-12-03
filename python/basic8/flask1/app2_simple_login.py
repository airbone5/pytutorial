from flask import Flask,render_template
from flask import request
app = Flask(__name__)


@app.route('/')
def home():
  return render_template('login.html')
  

@app.route('/login', methods=['GET', 'POST'])
def login():
  username = request.form.get('user_id')
  if not username=="123":
    return ("非使用者")
  else:
    return(render_template('welcome.html',user_id=username))

app.run(host='0.0.0.0',debug=True)
# if __name__ == '__main__':
#   app.run(host='0.0.0.0',debug=True)
  #app.run(host='0.0.0.0', port='8080', debug=True)
