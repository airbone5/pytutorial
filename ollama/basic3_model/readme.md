

[來源](https://github.com/ggerganov/llama.cpp/discussions/2948)
Install the huggingface_hub library:
```
pip install huggingface_hub
```
Create a Python script named download.py with the following content:
file name:`download.py`
```python
from huggingface_hub import snapshot_download
model_id="lmsys/vicuna-13b-v1.5"
snapshot_download(repo_id=model_id, local_dir="vicuna-hf",
                  local_dir_use_symlinks=False, revision="main")
```                  
Run the Python script:
```
python download.py
```
You should now have the model downloaded to a directory called
vicuna-hf. Verify by running:
```
ls -lash vicuna-hf
```

## Converting the model
Now it's time to convert the downloaded HuggingFace model to a GGUF model.
Llama.cpp comes with a converter script to do this.

1. Get the script by cloning the llama.cpp repo:
    ```
    git clone https://github.com/ggerganov/llama.cpp.git
    ```
1. Install the required python libraries:
    ```
    pip install -r llama.cpp/requirements.txt
    ```
1. Verify the script is there and understand the various options:
    ```
    python llama.cpp/convert_hf_to_gguf.py -h
    ```
1. Convert the HF model to GGUF model:
    ```
    python llama.cpp/convert_hf_to_gguf.py hf  --outfile vicuna-13b-v1.5.gguf  --outtype q8_0
    ```  
In this case we're also quantizing the model to 8 bit by setting
--outtype q8_0. Quantizing helps improve inference speed, but it can
negatively impact quality.
You can use --outtype f16 (16 bit) or --outtype f32 (32 bit) to preserve original
quality.

Verify the GGUF model was created:
```
ls -lash vicuna-13b-v1.5.gguf
```
## to ollama
```
FROM ./vicuna-13b-v1.5.gguf
```
接著，用戶需在 Ollama 平台上創建並運行這個模型。這涉及到下面的命令：

1. 使用命令創建模型：
    ```
    ollama create demo -f Modelfile
    ```
2. 運行剛剛創建的模型：
    ```
    ollama run demo
    ```