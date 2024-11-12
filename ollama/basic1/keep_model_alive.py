import requests

headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
}

data = '{\n  "model": "phi3:3.8b",\n  "keep_alive":-1\n}'

response = requests.post('http://localhost:11434/api/generate', headers=headers, data=data)