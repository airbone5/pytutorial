import requests

headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
}

data = '{\n  "model": "phi3:3.8b",\n  "keep_alive":-1\n}'
#http://localhost:11434/api/tags
response = requests.get('http://localhost:11434/api/tags')#,headers=headers)
print(response)