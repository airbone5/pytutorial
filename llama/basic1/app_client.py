from ollama import Client
client = Client(host='http://192.168.1.41:11434')
response = client.chat(model='llama3:8b', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])