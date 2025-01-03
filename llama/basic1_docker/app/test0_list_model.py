from ollama import Client
client = Client(host='http://192.168.1.41:11434')
rst=client.list()
list=rst['models']
#print(list)
for item in  list:
  print(item['name'])
