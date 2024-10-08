#import ollama
from ollama import Client

client = Client(host='http://192.168.1.41:11434')


response = client.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "system", "content": "你是保險法規助手,使用**繁體中文**回答。本法或法規指的是保險法。契約是保險契約。"},
        {"role": "user", "content": "本法所稱保險是甚麼"},
        {"role": "assistant", "content": "謂當事人約定，一方交付保險費於他方，他方對於因不可預料，或不可抗力之事故所致之損害，負擔賠償財物之行為"},
        {"role": "user", "content": "要保人是甚麼意思"},
        {"role": "assistant", "content": "要保人，指對保險標的具有保險利益，向保險人申請訂立保險契約，並負有交付保險費義務之人?"},
        {"role": "user", "content": "被保險人是甚麼意思"},
        {"role": "assistant", "content": "被保險人，指於保險事故發生時，遭受損害，享有賠償請求權之人；要保人亦得為被保險人"},
        {"role": "user", "content": "受益人是甚麼意思"},        
        {"role": "assistant", "content": "受益人指被保險人或要保人約定享有賠償請求權之人，要保人或被保險人均得為受益人"},
        {"role": "user", "content": "甚麼是保險業"},        
        {"role": "assistant", "content": "保險業，指依本法組織登記，以經營保險為業之機構。"},
        {"role": "user", "content": "受益人和被保險人的意思?"}        
    ]
)

print(response["message"]["role"])
print(response["message"]["content"])