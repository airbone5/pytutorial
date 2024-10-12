import csv
# 😒不使用encoding會遇到甚麼錯誤
with open('name.csv', newline='', encoding="utf8") as csvfile:    
    reader = csv.DictReader(csvfile)
    
    for row in reader:    
        print(row['學號'], row['姓名'])