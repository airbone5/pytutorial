#import ollama
from ollama import Client

client = Client(host='http://192.168.1.41:11434')

q1 = "What is 2+2?"
a1 = "2 + 2 equals 4. This is a basic arithmetic operation where you are " \
        "adding the number 2 to another number 2, resulting in 4."

q2 = "Add 3 onto the last answer. What is it now?"

response = client.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "user", "content": q1},
        {"role": "assistant", "content": a1},
        {"role": "user", "content": q2}
    ]
)

print(response["message"]["role"])
print(response["message"]["content"])