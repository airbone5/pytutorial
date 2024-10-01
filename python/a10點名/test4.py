
import csv

def readname():
# 😒不使用encoding會遇到甚麼錯誤
  with open('name.csv', newline='', encoding="utf8") as csvfile:    
      reader = csv.DictReader(csvfile)
      result=[]
      for row in reader:    
         result.append({'學號':row['學號'],'姓名':row['姓名']})
      return result   
  
x=readname()
print(x)
