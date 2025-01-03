#import ollama
from ollama import Client

client = Client(host='http://192.168.1.41:11434')


response = client.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "system", "content": "你是保險法規助手,使用**繁體中文**回答。本法或法規指的是保險法。契約是保險契約。"},
        {"role": "user", "content": "甚麼是保險?"},
        {"role": "user", "content": "受益人和被保險人的意思?"} 
    ]
)

print(response["message"]["role"])
print(response["message"]["content"])