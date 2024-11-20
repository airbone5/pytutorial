---
title: demo
description: docker log
weight: 300
---
In this tutorial, we'll walk through the process of training a language model using the Llama model architecture and the Transformers library.
1. Installing the Required Libraries
We'll start by installing the necessary libraries using pip:



```python
%pip install -q datasets accelerate evaluate trl accelerate transformers jinja2
```

    Note: you may need to restart the kernel to use updated packages.
    

    
    [notice] A new release of pip is available: 24.2 -> 24.3.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
    

2. Logging into Hugging Face Hub
Next, we'll log into the Hugging Face Hub to access the required models and datasets:


```python
from huggingface_hub import notebook_login

notebook_login()
```


    VBox(children=(HTML(value='<center> <img\nsrc=https://huggingface.co/front/assets/huggingface_logo-noborder.sv…


3. Loading the Necessary Libraries and Models
We'll import the required libraries and load the Llama model and tokenizer:

this part is pretty complicated, so stay with me.



```python

from datasets import load_dataset

dataset = load_dataset("your_dataset_name", split="train") # load the dataset

```


    ---------------------------------------------------------------------------

    DatasetNotFoundError                      Traceback (most recent call last)

    Cell In[8], line 3
          1 from datasets import load_dataset
    ----> 3 dataset = load_dataset("your_dataset_name", split="train") # load the dataset
    

    File c:\pywork\ollama\basic5_torch\prj\Lib\site-packages\datasets\load.py:2132, in load_dataset(path, name, data_dir, data_files, split, cache_dir, features, download_config, download_mode, verification_mode, keep_in_memory, save_infos, revision, token, streaming, num_proc, storage_options, trust_remote_code, **config_kwargs)
       2127 verification_mode = VerificationMode(
       2128     (verification_mode or VerificationMode.BASIC_CHECKS) if not save_infos else VerificationMode.ALL_CHECKS
       2129 )
       2131 # Create a dataset builder
    -> 2132 builder_instance = load_dataset_builder(
       2133     path=path,
       2134     name=name,
       2135     data_dir=data_dir,
       2136     data_files=data_files,
       2137     cache_dir=cache_dir,
       2138     features=features,
       2139     download_config=download_config,
       2140     download_mode=download_mode,
       2141     revision=revision,
       2142     token=token,
       2143     storage_options=storage_options,
       2144     trust_remote_code=trust_remote_code,
       2145     _require_default_config_name=name is None,
       2146     **config_kwargs,
       2147 )
       2149 # Return iterable dataset in case of streaming
       2150 if streaming:
    

    File c:\pywork\ollama\basic5_torch\prj\Lib\site-packages\datasets\load.py:1853, in load_dataset_builder(path, name, data_dir, data_files, cache_dir, features, download_config, download_mode, revision, token, storage_options, trust_remote_code, _require_default_config_name, **config_kwargs)
       1851     download_config = download_config.copy() if download_config else DownloadConfig()
       1852     download_config.storage_options.update(storage_options)
    -> 1853 dataset_module = dataset_module_factory(
       1854     path,
       1855     revision=revision,
       1856     download_config=download_config,
       1857     download_mode=download_mode,
       1858     data_dir=data_dir,
       1859     data_files=data_files,
       1860     cache_dir=cache_dir,
       1861     trust_remote_code=trust_remote_code,
       1862     _require_default_config_name=_require_default_config_name,
       1863     _require_custom_configs=bool(config_kwargs),
       1864 )
       1865 # Get dataset builder class from the processing script
       1866 builder_kwargs = dataset_module.builder_kwargs
    

    File c:\pywork\ollama\basic5_torch\prj\Lib\site-packages\datasets\load.py:1717, in dataset_module_factory(path, revision, download_config, download_mode, dynamic_modules_path, data_dir, data_files, cache_dir, trust_remote_code, _require_default_config_name, _require_custom_configs, **download_kwargs)
       1715     raise ConnectionError(f"Couldn't reach the Hugging Face Hub for dataset '{path}': {e1}") from None
       1716 if isinstance(e1, (DataFilesNotFoundError, DatasetNotFoundError, EmptyDatasetError)):
    -> 1717     raise e1 from None
       1718 if isinstance(e1, FileNotFoundError):
       1719     if trust_remote_code:
    

    File c:\pywork\ollama\basic5_torch\prj\Lib\site-packages\datasets\load.py:1643, in dataset_module_factory(path, revision, download_config, download_mode, dynamic_modules_path, data_dir, data_files, cache_dir, trust_remote_code, _require_default_config_name, _require_custom_configs, **download_kwargs)
       1639     raise DatasetNotFoundError(
       1640         f"Revision '{revision}' doesn't exist for dataset '{path}' on the Hub."
       1641     ) from e
       1642 except RepositoryNotFoundError as e:
    -> 1643     raise DatasetNotFoundError(f"Dataset '{path}' doesn't exist on the Hub or cannot be accessed.") from e
       1644 try:
       1645     dataset_script_path = api.hf_hub_download(
       1646         repo_id=path,
       1647         filename=filename,
       (...)
       1650         proxies=download_config.proxies,
       1651     )
    

    DatasetNotFoundError: Dataset 'your_dataset_name' doesn't exist on the Hub or cannot be accessed.


Here, we'll get the corpus to past to the tokenizer


```python
def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

training_corpus = get_training_corpus()
```

The base tokenizer is up to you, I'm using a blank one, but a lot of people opt for different ones, such as gpt2.


```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(
    training_corpus,
    vocab_size=3200,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>", "<|user|>", "<|bot|>", "<|end|>"] # you can pick the last two or three, as you'll see next
)
```

Next, we'll define the tokenizer special tokens and chat template.


```python


from transformers import PreTrainedTokenizerFast

special_tokens = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    "additional_special_tokens": ["<|user|>", "<|bot|>", "<|end|>"] # same here
}
tokenizer.add_special_tokens(special_tokens)

tokenizer.user_token_id = tokenizer.convert_tokens_to_ids("<|user|>") # here
tokenizer.assistant_token_id = tokenizer.convert_tokens_to_ids("<|bot|>") # too

chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '<|end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|bot|>\n' + message['content'] + '<|end|>\n' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}{{ eos_token }}" # this is where you define the chat template, so you can go crazy here. Something a lot of people do for whatever reason is add seamingly random newline characters

tokenizer.chat_template = chat_template

```

Now, finally, we'll define the model.


```python
from transformers import LlamaConfig, LlamaForCausalLM

print(tokenizer.apply_chat_template([{"role": "user", "content": "Why is the sky blue?"}, {"role": "assistant", "content": "Due to rayleigh scattering."}], tokenize=False)) # test to see if the chat template worked

config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=512,
    intermediate_size=1024,
    num_hidden_layers=8,
    num_attention_heads=8,
    max_position_embeddings=512,
    rms_norm_eps=1e-6,
    initializer_range=0.02,
    use_cache=True,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    tie_word_embeddings=False,
)

model = LlamaForCausalLM(config)
```

4. Formatting the Dataset
We'll define a function to format the prompts in the dataset and map the dataset:


```python
def format_prompts(examples):
    """
    Define the format for your dataset
    This function should return a dictionary with a 'text' key containing the formatted prompts.
    """
    pass

dataset = dataset.map(format_prompts, batched=True)

print(dataset['text'][2]) # Check to see if the fields were formatted correctly
```

5. Setting Up the Training Arguments
Define the training args:


```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="your_output_dir",
    num_train_epochs=4, # replace this, depending on your dataset
    per_device_train_batch_size=16,
    learning_rate=1e-4,
    optim="sgd" # sgd, my beloved
)
```

6. Creating the Trainer
We'll create an instance of the SFTTrainer from the trl library:


```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=dataset,
    dataset_text_field='text',
    max_seq_length=512
)
```

7. Training the Model
Finally, we'll start the training process:


```python
trainer.train()
```

8. Pushing the Trained Model to Hugging Face Hub
After the training is complete, you can push the trained model to the Hugging Face Hub using the following command:


```python
trainer.push_to_hub()
```

This will upload the model to your Hugging Face Hub account, making it available for future use or sharing.

That's it!
