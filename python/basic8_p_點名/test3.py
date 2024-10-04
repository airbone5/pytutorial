from flask import Flask,render_template,redirect,url_for,jsonify


#app = Flask(__name__)
app = Flask(__name__, static_url_path='/static') 

@app.route('/')
def home():
# 方法1:直接  
#   return {
#         "user": "John Doe",
#   }
# 方法2: 
  return jsonify(username="xx",email="dd")
# 測試1
#   bar = '<body>好</body>'      
#   response = make_response(bar)
#   response.headers['Content-Type'] = 'text/xml; charset=utf-8'
#   return response
 

if __name__ == '__main__':
  app.run(debug=True)

