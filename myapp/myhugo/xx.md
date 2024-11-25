```python
%pip uninstall -y matplotlib
```

    Found existing installation: matplotlib 3.9.2
    Uninstalling matplotlib-3.9.2:
      Successfully uninstalled matplotlib-3.9.2
    Note: you may need to restart the kernel to use updated packages.
    


```python
%pip install matplotlib
```

    Collecting matplotlib
      Using cached matplotlib-3.9.2-cp312-cp312-win_amd64.whl.metadata (11 kB)
    Requirement already satisfied: contourpy>=1.0.1 in c:\pywork\myapp\myhugo\prj\lib\site-packages (from matplotlib) (1.3.1)
    Requirement already satisfied: cycler>=0.10 in c:\pywork\myapp\myhugo\prj\lib\site-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in c:\pywork\myapp\myhugo\prj\lib\site-packages (from matplotlib) (4.55.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in c:\pywork\myapp\myhugo\prj\lib\site-packages (from matplotlib) (1.4.7)
    Requirement already satisfied: numpy>=1.23 in c:\pywork\myapp\myhugo\prj\lib\site-packages (from matplotlib) (2.1.3)
    Requirement already satisfied: packaging>=20.0 in c:\pywork\myapp\myhugo\prj\lib\site-packages (from matplotlib) (24.1)
    Requirement already satisfied: pillow>=8 in c:\pywork\myapp\myhugo\prj\lib\site-packages (from matplotlib) (11.0.0)
    Requirement already satisfied: pyparsing>=2.3.1 in c:\pywork\myapp\myhugo\prj\lib\site-packages (from matplotlib) (3.2.0)
    Requirement already satisfied: python-dateutil>=2.7 in c:\pywork\myapp\myhugo\prj\lib\site-packages (from matplotlib) (2.9.0.post0)
    Requirement already satisfied: six>=1.5 in c:\pywork\myapp\myhugo\prj\lib\site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)
    Using cached matplotlib-3.9.2-cp312-cp312-win_amd64.whl (7.8 MB)
    Installing collected packages: matplotlib
    Successfully installed matplotlib-3.9.2
    Note: you may need to restart the kernel to use updated packages.
    

    
    [notice] A new release of pip is available: 24.2 -> 24.3.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
    


```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

fruits = ['apple', 'blueberry', 'cherry', 'orange']
counts = [40, 100, 30, 55]
bar_labels = ['red', 'blue', '_red', 'orange']
bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

ax.set_ylabel('fruit supply')
ax.set_title('Fruit supply by kind and color')
ax.legend(title='Fruit color')

plt.show()
```


    
![png](output_2_0.png)
    


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

    Cell In[2], line 1
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
