from flask import Flask,render_template,redirect,url_for,jsonify,make_response
import csv
import pyqrcode
from pyqrcode import QRCode

#app = Flask(__name__)
app = Flask(__name__, static_url_path='/static') 

app.config["JSON_AS_ASCII"] = False
app.config["JSONIFY_MIMETYPE"] = "application/json; charset=utf-8"


def readname():
# 😒不使用encoding會遇到甚麼錯誤
  with open('name.csv', newline='', encoding="utf8") as csvfile:    
      reader = csv.DictReader(csvfile)
      result='<table>'
      for row in reader:    
         result=result+ '<tr><td>'+row['學號'] + '</td><td>'+row['姓名']+'</td></tr>';
      result=result+'</table>'
      return result   
  

@app.route('/')
def home():
  response = make_response(readname()) 
  return  response  
  
svghead='<svg height="210" width="400" xmlns="http://www.w3.org/2000/svg">'
svgtail='</svg>'  
 
@app.route("/denming/<name>") #, methods=["POST"])
def add_course_to_student(name):
  plusurl=pyqrcode.create(name) 
  pos='./temp/'+name+".svg"
  plusurl.svg(pos, scale=8)
  txt=open(pos).read().replace('<?xml version="1.0" encoding="UTF-8"?>','') 
  return make_response(txt)


 

if __name__ == '__main__':
  app.run(debug=True)

