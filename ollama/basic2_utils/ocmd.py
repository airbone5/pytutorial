import argparse
import sys
import requests
import json



def list_model():
    response = requests.get('http://localhost:11434/api/tags')
    rst=response.json()    
    for item in rst['models']:
        print(item['name'])
def forever():
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    data = '{\n  "model": "phi3:3.8b",\n  "keep_alive":-1\n}'
    response = requests.post('http://localhost:11434/api/generate', headers=headers, data=data)
def unload():
# curl http://localhost:11434/api/generate -d '{
#   "model": "llama3.2",
#   "keep_alive": 0
# }'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    data = '{\n  "model": "phi3:3.8b",\n  "keep_alive":0\n}'
    response = requests.post('http://localhost:11434/api/generate', headers=headers, data=data)   

def show():
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    data = '{"name": "phi3:3.8b"}'
    response = requests.post('http://localhost:11434/api/show', headers=headers, data=data)    
    #json_formatted_str = json.dumps(response.json(), indent=2)
    #print(json_formatted_str)    
    print(response.json()['modelfile'])
   

parser = argparse.ArgumentParser()
parser.add_argument("command") # 命令例如show,list,unload
parser.add_argument("model",  nargs='?',default="phi3:3.8b",help="模型名稱預設:phi3:3.8b")# 字串
#parser.add_argument("arg3",  nargs='?',default=0, help="第 3 個參數", type=int)
#parser.add_argument("-v", "--verbose", help="比較多的說明", action="store_true")

options = parser.parse_args()    


if options.command == "list":
    list_model()
elif options.command == "forever":
    forever()
elif options.command == "show":
    print(options.model)
    show()    
elif options.command == "unload":
    unload()        
else:
    print('give me command:list, forever,show,unload')