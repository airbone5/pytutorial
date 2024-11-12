from langchain_community.llms import Ollama
llm = Ollama(model='llama3', base_url='http://192.168.1.41') #phi3

response=llm.invoke("hello, tell me a programmer jokes.")
print(response)