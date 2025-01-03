## [安裝pytorch](https://pytorch.org/get-started/locally/)
- only cpu
```
pip3 install torch torchvision torchaudio
```

# Ollama
## 參考
- [ollama example tinyllama(Github)](https://ollama.com/library/tinyllama:1.1b)

## 環境
- 沒有nvidia 只有CPU(在虛擬機上)
- 測試可以用的MODEL有
   ```
   ollama run llama3:8b
   ## 問下面的問題
   ##can i run ollama without gpu
   ```

[ok](https://github.com/owainlewis/awesome-artificial-intelligence?tab=readme-ov-file)


## curl 
[demo by openai](https://ollama.com/blog/openai-compatibility)
[ollama api](https://github.com/ollama/ollama/blob/main/docs/api.md)

- chat
```
curl http://192.168.1.41:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Why is the sky blue?"
}'
```
```bash
curl http://192.168.1.41:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Why is the sky blue?",
  "stream": false,
  "keep_alive":-1
}'

curl http://192.168.1.41:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "keep_alive":-1
}'
```
上面curl([可以參考python](./basic1/keep_model_alive.py))是否執行成功,可以利用`ollama ps`查看,例如
```
# ollama ps
NAME           ID              SIZE      PROCESSOR    UNTIL              
llama3.1:8b    f66fc8dc39ea    6.7 GB    100% GPU     4 minutes from now    

# curl http://192.168.1.41:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "keep_alive":-1
}'

# ollama ps
NAME           ID              SIZE      PROCESSOR    UNTIL   
llama3.1:8b    f66fc8dc39ea    6.7 GB    100% GPU     Forever  
```

unload model
```
curl http://192.168.1.41:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "keep_alive": 0
}
curl http://192.168.1.41:11434/api/generate -d '{
  "model": "llama2:7b",
  "keep_alive": 0
}
```
## why python
[dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

[
   {
      "from":"system",
      "value":"You are an advanced AI assistant created to perform sentiment analysis on text. Your task is to carefully read the text and analyze the sentiment it expresses towards the potential future stock value of any company mentioned. Analyze the sentiment of this text and respond with the appropriate JSON:"
   },
   {
      "from":"human",
      "value":"@USAirways Sadly I'm currently booked on one of your flights tomorrow that I will be changing if I can."
   },
   {
      "from":"gpt",
      "value":"{\"reasoning\": \"The sentiment is negative based on the content of the tweet.\", \"sentiment\": -1.0, \"confidence\": 0.68}"
   }
]
[
   {
      "from":"system",
      "value":"You are an advanced AI assistant created to perform sentiment analysis on text. Your task is to carefully read the text and analyze the sentiment it expresses towards the potential future stock value of any company mentioned. Analyze the sentiment of this text and respond with the appropriate JSON:"
   },
   {
      "from":"human",
      "value":"As of July 2 , 2007 , the market cap segments will be updated according to the average price in May 2007 ."
   },
   {
      "from":"gpt",
      "value":"{\"reasoning\": \"The sentiment is neutral based on the financial context of the sentence.\", \"sentiment\": 0.0, \"confidence\": 0.4}"
   }
]


[ { "from": "system", "value": "You are an advanced AI assistant created to perform sentiment analysis on text. Your task is to carefully read the text and analyze the sentiment it expresses towards the potential future stock value of any company mentioned. Analyze the sentiment of this text and respond with the appropriate JSON:" }, { "from": "human", "value": "@USAirways Sadly I'm currently booked on one of your flights tomorrow that I will be changing if I can." }, { "from": "gpt", "value": "{\"reasoning\": \"The sentiment is negative based on the content of the tweet.\", \"sentiment\": -1.0, \"confidence\": 0.68}" } ]

[ { "from": "system", "value": "You are an advanced AI assistant created to perform sentiment analysis on text. Your task is to carefully read the text and analyze the sentiment it expresses towards the potential future stock value of any company mentioned. Analyze the sentiment of this text and respond with the appropriate JSON:" }, { "from": "human", "value": "As of July 2 , 2007 , the market cap segments will be updated according to the average price in May 2007 ." }, { "from": "gpt", "value": "{\"reasoning\": \"The sentiment is neutral based on the financial context of the sentence.\", \"sentiment\": 0.0, \"confidence\": 0.4}" } ]