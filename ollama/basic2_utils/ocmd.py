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
    json_formatted_str = json.dumps(response.json(), indent=2)
    print(json_formatted_str)    

   
    
option=sys.argv[1]

if option == "list":
    list_model()
elif option == "forever":
    forever()
elif option == "show":
    show()    
elif option == "unload":
    unload()        
else:
    print('give me command:list, forever,show,unload')
# elif lang == "Python":
#     return "You can become a Data Scientist"
# elif lang == "Solidity":
#     return "You can become a Blockchain developer."
# elif lang == "Java":
#     return "You can become a mobile app developer"