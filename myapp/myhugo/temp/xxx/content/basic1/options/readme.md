---
title: readme
description: docker log
weight: 300
---
 
## 甚麼是sys.argv
新增檔案hello.py,並將下面的內容貼上,然後執行
```python
# demo python hello.py --option 1
import sys
 

def main(): #Note1: def main1()
    # standard - load "args" list with cmd-line-args
    args = sys.argv[0:] #冒號後面是空的代表從0到最後
    print(args)



if __name__ == '__main__':  
    main() # 如果Note1的函數是main1,那這裡就是main1(),不是一定要main()
```       
執行輸出:
```
start
['c:\\Users\\linchao\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\ipykernel_launcher.py', '--f=c:\\Users\\linchao\\AppData\\Roaming\\jupyter\\runtime\\kernel-v31f52aa1600e6bb88d599e1fcd037bc6640b1fb1b.json']
```
## 利用其他套件
上面的方法利用的是內建套件`sys` 也可以利用`argparse`, `click`

### 套件argparse
- [參考argparse](https://docs.python.org/zh-tw/3/howto/argparse.html)
[](程式碼
)
```bash
python arg_click1.py
python arg_click1.py --v
```

```python
#demo python arg_click1.py
import argparse
parser = argparse.ArgumentParser()

```    


```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("arg1") # 字串
parser.add_argument("arg2",  nargs='?',default="b",help="第 2 個參數")# 字串
parser.add_argument("arg3",  nargs='?',default=0, help="第 3 個參數", type=int)
parser.add_argument("-v", "--verbose", help="比較多的說明", action="store_true")

args = parser.parse_args()
if args.verbose:
    print("這是測試結果")

print(f"第 1 個參數：{args.arg1:^10},type={type(args.arg1)}")
print(f"第 2 個參數：{args.arg2:^10},type={type(args.arg2)}")
print(f"第 3 個參數：{args.arg3:^10},type={type(args.arg3)}")
```

### 套件click

```python
import click

@click.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name',
              help='The person to greet.')
def hello(count, name):
    """Simple program that greets NAME for a total of COUNT times."""
    for x in range(count):
        click.echo(f"Hello {name}!")

if __name__ == '__main__':
    hello()
```