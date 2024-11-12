import pandas
import openpyxl
from datetime import datetime

df=pandas.read_excel('test.xlsx',sheet_name='工作表1', skiprows=6)

# print(type(df))
# print(df.head())
 

# <class 'pandas.core.frame.DataFrame'> 👍mydataset 已經是dataframe
#     Unnamed: 0  Unnamed: 1    項次    姓名           學號  Unnamed: 5  ...  Unnamed: 21  Unnamed: 22    小考 Unnamed: 24 Unnamed: 25 Unnamed: 26
# 0          NaN         NaN    1   洪欣佩   C112130101   C112130101  ...          NaN          NaN   3.0        85.0       100.0         NaN 
# 1          NaN         NaN    2   沙旻慧   C112130102   C112130102  ...          NaN          NaN   4.0        90.0       100.0         NaN 
crst=[]
xx=df.columns
for  col in xx:
    
    if type(col).__name__=='datetime':
        #print(col.strftime('%Y%m%d'))
        yy=col.strftime('%Y%m%d')
        crst=crst+[yy]
    else:
        crst=crst+[col]
         
#df.columns = [col.replace(":", "_") for col in df.columns]
df.columns=crst

# 清資料:把Unnamed 的欄位去掉
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print(df.head())

#     result = op(self._values)
#              ^^^^^^^^^^^^^^^^
# TypeError: bad operand type for unary ~: 'float'