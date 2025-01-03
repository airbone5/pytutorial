子目錄說明(tree -d -x)
```
.
├── fromhf   demo hugginface to gguf ./fromhf
├── llama.cpp 工具
└── prj
```

### 常用指令 
ollama show --modelfile qwen2.5:0.5b


## 其他參考
- [Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#template)
- [Ollama - Building a Custom Model](https://unmesh.dev/post/ollama_custom_model/)
  ```
  # Modelfile for creating an API security assistant
  # Run `ollama create api-secexpert -f ./Modelfile` and then `ollama run api-secexpert` and enter a topic

  FROM codellama
  PARAMETER temperature 1

  SYSTEM """
  You are a senior API developer expert, acting as an assistant. 
  You offer help with API security topics such as: Secure Coding practices, 
  API security, API endpoint security, OWASP API Top 10. 
  You answer with code examples when possible.
  """
  ```


- [Modelfile example](https://medium.com/@sudarshan-koirala/ollama-huggingface-8e8bc55ce572) 用到[這個](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

  ```txt
  # Modelfile
  FROM "./capybarahermes-2.5-mistral-7b.Q4_K_M.gguf"

  PARAMETER stop "<|im_start|>"
  PARAMETER stop "<|im_end|>"

  TEMPLATE """
  <|im_start|>system
  {{ .System }}<|im_end|>
  <|im_start|>user
  {{ .Prompt }}<|im_end|>
  <|im_start|>assistant
  """
  ```