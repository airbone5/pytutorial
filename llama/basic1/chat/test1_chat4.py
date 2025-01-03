#import ollama
from ollama import Client

client = Client(host='http://192.168.1.41:11434')


response = client.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "system", "content": "你是保險法規助手,使用**繁體中文**回答。本法或法規指的是保險法。契約是保險契約。"},
        {"role": "user", "content": "本法所稱保險是甚麼"},
        {"role": "assistant", "content": "謂當事人約定，一方交付保險費於他方，他方對於因不可預料，或不可抗力之事故所致之損害，負擔賠償財物之行為"},
        {"role": "user", "content": "甚麼是保險?"}
    ]
)

print(response["message"]["role"])
print(response["message"]["content"])