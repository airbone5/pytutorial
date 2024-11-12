---
title: tutorial
description: docker log
weight: 300
---
# 輸出輸入


```python
# 字串
print("hello")
# 字串變數
prompt="hello"
print(prompt)
#字串運算
print(prompt+"hello")
```

    hello
    hello
    hellohello
    


```python
f = open("demofile3.txt", "w") # open("demo.txt","w",encoding='utf8')
f.write("Woops! I have deleted the content!")
f.close()

```

### 輸入


```python
aname = input(r"你的姓名: ")
print(f"剛剛輸入的字串: {aname}")
```

    Files in the directory: justme
    

## 甚麼是import
教材位置: 子目錄`packages`[這裡](./packages/套件.md)

## 常用資料結構

- list and tuple[參考](https://utrustcorp.com/python_list_tuple/)


```python
['a']+['b']
#['a']+['b']-['a'] ❌不支援減號

```




    ['a', 'b']




```python
['a']*5 # ['a']/['a']不支援
```




    ['a', 'a', 'a', 'a', 'a']




```python
path='dd' / 'dd'
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[6], line 1
    ----> 1 path='dd' / 'dd'
    

    TypeError: unsupported operand type(s) for /: 'str' and 'str'



```python
x=[1,2,3,4]
print(x[:-1])
x[-1]
```

## range


```python
for item in range(1,5):
    print(item)
```

    1
    2
    3
    4
    

## 甚麼是sys.argv

新增檔案`hello.py`,並將下面的內容貼上,然後執行 
```
python hello.py --option 1
```


```python
import sys
 

def main(): #Note1: def main1()
    # standard - load "args" list with cmd-line-args
    args = sys.argv[0:]
    print(args)
   

# Python boilerplate.
if __name__ == '__main__':
    print('start')
    main() # 如果Note1的函數是main1,那這裡就是main1(),不是一定要main()

```

    start
    ['c:\\Users\\linchao\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\ipykernel_launcher.py', '--f=c:\\Users\\linchao\\AppData\\Roaming\\jupyter\\runtime\\kernel-v31f52aa1600e6bb88d599e1fcd037bc6640b1fb1b.json']
    

續前

在Notebook中執行 ❓哪裡不一樣?


```python
import sys
print(sys.argv)
```

    ['c:\\Users\\linchao\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\ipykernel_launcher.py', '--f=c:\\Users\\linchao\\AppData\\Roaming\\jupyter\\runtime\\kernel-v3e90487ec4d919f659b93707db636429a7b31ed65.json']
    

[參考argparse](https://docs.python.org/zh-tw/3/howto/argparse.html)
