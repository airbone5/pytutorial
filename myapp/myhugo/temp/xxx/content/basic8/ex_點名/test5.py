from flask import Flask,render_template,redirect,url_for,jsonify,make_response
import csv

#app = Flask(__name__)
app = Flask(__name__, static_url_path='/static') 
#⭕😒無效
app.config["JSON_AS_ASCII"] = False
app.config["JSONIFY_MIMETYPE"] = "application/json; charset=utf-8"

def readname():
# 😒不使用encoding會遇到甚麼錯誤
  with open('name.csv', newline='', encoding="utf8") as csvfile:    
      reader = csv.DictReader(csvfile)
      result=[]
      for row in reader:    
         result.append({'no':row['學號'],'name':row['姓名']})
      return result   
  

@app.route('/')
def home():
  
  response = jsonify(readname()) 
  return  response  

 

if __name__ == '__main__':
  app.run(debug=True)

