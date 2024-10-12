import csv
"""
iris.csv的資料
150,4,setosa,versicolor,virginica
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
"""
handle=open('../dataset/iris.csv')
rst=csv.reader(handle)
# # 轉為list
# print(list(rst))

# # 轉為tuple
# print(tuple(rst))

# # 轉為dict❌
# x=dict(rst)
reader = csv.DictReader(handle)
print(reader)
print(reader.fieldnames)
