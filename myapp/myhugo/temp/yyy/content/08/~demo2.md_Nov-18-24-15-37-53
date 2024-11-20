---
title: demo2
description: docker log
weight: 300
---
 [本文參考](https://www.linkedin.com/pulse/customizing-llama-3-guide-fine-tuning-your-dataset-ali-farahani-hllke)
 
 Llama-3 is one of the latest LLMs released by Meta under an open-source license. Its impressive performance in different tasks such as text-generation, translation, question answering, etc., position it as a strong competitor to closed-source models like GPT-4.

 The key advantages of Llama-3 include:

 - Llama-3 is an open-source LLM, you can use it and fine-tune it to your needs.

 - Llama-3 comes in two sizes: 8billion and 70billion parameters. The 70B version competes effectively with closed-source models like GPT-4.

 - Llama-3 is multi-lingual, means it understands different languages such as English, French, Persian, and many more.

 - While the Llama-3 is open-source, accessing it requires approval from Meta's Llama-3 AI team. This may present some limitations or barriers for certain users :(.

 These features make Llama-3 an ideal choice for users who want to adapt it to their specific tasks. This guide will walk you trough to use Llama-3 with 8 billion parameters, and fine-tune it with your custom dataset step-by-step. Note that we don’t go through the technical details of Llama-3 and transformers, we simply utilize it.

 We use hugging face's transformers library in this text. The python packages you need are as follows:

 - Transformers: HuggingFace Transformer Library.

 - trl: HuggingFace Transformer Reinforcement Library (we use trl for fine-tuning Llama-3).

 - datasets: HuggingFace community-driven open-source library for datasets.

 - peft: HuggingFace Parameter-Efficient Fine-Tuning (peft is useful in cases we don't have powerful GPUs for fine-tuning)

 To install these packages you can simply use pip command:


```python
! pip install transformers trl datasets peft        
```

 # Defining Llama-3 model

 The first step is to create a Llama-3 model, download and load its weights. As mentioned earlier, official Llama-3 model is not publicly available. To get access to it you need to sign a form provided in this address (https://huggingface.co/meta-llama/Meta-Llama-3-8B).

 You can also use Llama-3 models shared by other users or organizations. These models are based on the official Llama-3, but they've been fine-tuned on specific datasets, potentially altering their capabilities.

 You can see a list of available Llama-3 models shared by community in HuggingFace models page(https://huggingface.co/models?sort=trending&search=llama-3).

 We use "NousResearch/Meta-Llama-3-8B-Instruct" model.

 The following codes creates and loads the model. If your computer has a CUDA capable GPU and the drivers are installed correctly, it will utilize your GPU.

 Quantization_config is used to reduce the memory needed for loading the model. The quantized version requires around 6GB of GPU memory to load. If you don’t use quantization (by removing quantization_config=quantization_config line) the memory needed for loading the model increases to around 12GB.


```python
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4"
)

# 如果不指定cache_dir 可以在%userprofile%/.cache中

# 下面指定GPU的方法,我覺得沒用,但是不確定
#device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model = AutoModelForCausalLM.from_pretrained(
#     "NousResearch/Meta-Llama-3-8B-Instruct", 
#     quantization_config=quantization_config, 
#     device_map= device, #"cuda:0" or "auto", 我覺得沒用到GPU
#     cache_dir='../../pretrain/'
    
# )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Meta-Llama-3-8B-Instruct", 
    quantization_config=quantization_config, 
    cache_dir='../../pretrain/'
    
).to(device)



tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/Meta-Llama-3-8B-Instruct",
    cache_dir='../../pretrain/'
)       

```

    `low_cpu_mem_usage` was None, now default to True since model is quantized.
    


    Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]


    You shouldn't move a model that is dispatched using accelerate hooks.
    


```python
model.config
```




    LlamaConfig {
      "_attn_implementation_autoset": true,
      "_name_or_path": "NousResearch/Meta-Llama-3-8B-Instruct",
      "architectures": [
        "LlamaForCausalLM"
      ],
      "attention_bias": false,
      "attention_dropout": 0.0,
      "bos_token_id": 128000,
      "eos_token_id": 128009,
      "head_dim": 128,
      "hidden_act": "silu",
      "hidden_size": 4096,
      "initializer_range": 0.02,
      "intermediate_size": 14336,
      "max_position_embeddings": 8192,
      "mlp_bias": false,
      "model_type": "llama",
      "num_attention_heads": 32,
      "num_hidden_layers": 32,
      "num_key_value_heads": 8,
      "pretraining_tp": 1,
      "quantization_config": {
        "_load_in_4bit": true,
        "_load_in_8bit": false,
        "bnb_4bit_compute_dtype": "float32",
        "bnb_4bit_quant_storage": "uint8",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": true,
        "llm_int8_enable_fp32_cpu_offload": false,
        "llm_int8_has_fp16_weight": false,
        "llm_int8_skip_modules": null,
        "llm_int8_threshold": 6.0,
        "load_in_4bit": true,
        "load_in_8bit": false,
        "quant_method": "bitsandbytes"
      },
      "rms_norm_eps": 1e-05,
      "rope_scaling": null,
      "rope_theta": 500000.0,
      "tie_word_embeddings": false,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.46.1",
      "use_cache": true,
      "vocab_size": 128256
    }



By running this block of code, the model is downloaded from huggingface's model hub and copied to `~/.cache/huggingface/` directory (in Linux machines). Then the downloaded weights are loaded into your computers memory (based on device=gpu | cpu).

## Using Llama-3

Now you can use Llama-3. For using a LLM you first need to know its prompt template.

Prompt template is a reusable structure that helps users craft effective prompts. These templates act as a blueprint, providing a framework to guide the LLM towards generating the desired output.

Prompt templates typically include placeholders or slots where users can insert unique input data or instructions. This allows for a high degree of customization while maintaining the overall structure that has been optimized for the LLM's capabilities.

The prompt template of Llama-3 is as follows (https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/):
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{CONTEXT}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{QUESTION}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{CONTEXT} is additional information or background provided to the model alongside the main question.

{QUESTION} is the question you ask from the LLM.
```
Example:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an Artificial Intelligence assistant. Answer the questions in an academic style.

<|eot_id|><|start_header_id|>user<|end_header_id|>

What are some advantages of ReLU activation function?

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```
Now you can define a prompt and pass it to Llama-3 for inference:



```python
prompt="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

你是保險法規助理。用中文回答問題。

<|eot_id|><|start_header_id|>user<|end_header_id|>

甚麼是受益人?

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
tokenized_chat = tokenizer.encode(prompt, return_tensors="pt")

tokenized_chat = tokenized_chat.to(device)

generate_ids = model.generate(tokenized_chat, max_new_tokens=256)

print(tokenizer.decode(generate_ids[0]))

```

    The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    c:\pywork\ollama\basic5_torch\prj\Lib\site-packages\bitsandbytes\nn\modules.py:452: UserWarning: Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default). This will lead to slow inference or training speed.
      warnings.warn(
    

    <|begin_of_text|><|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    你是保險法規助理。用中文回答問題。
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    
    甚麼是受益人?
    
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    在保險法規中，受益人（Beneficiary）是指被繼承人或被賠償的個人或機構，受益於保險契約中的保險金或賠償金。也就是說，受益人是指被指定或指定的個人或機構，將在保險人死亡或傷亡時或因為保險事故而獲得保險金或賠償金的對象。<|eot_id|>
    


 Result:

 The result of running the codes in a ipython session in vscode.

 As can be seen from the code block, we first organize our prompt in the correct tramplate. Then we use the tokenizer to convert the prompt text into a list of token_ids. Finally, we pass the tokenized prompt to model.generate() method. Tokenizer.decode() is used to convert token_ids to a string so we can print it to see the result.

## Fine-tuning Llama-3

 Fine-tuning a HuggingFace model using the Transformers Reinforcement Learning (trl) library is a very simple process. The key requirement is to create a dataset that is compatible with the trl library. The dataset can be structured as a list of dictionaries, where each dictionary item has a "text" key. This "text" key should contain the input text that you want to use for fine-tuning the model.

 See the example bellow:

 PROMPT 甚麼是保險
RESPONSE 保險指當事人約定，一方交付保險費於他方，他方對於因不可預料，或不可抗力之事故所致之損害，負擔賠償財物之行為
PROMPT 甚麼是保險人
RESPONSE 保險人指經營保險事業之各種組織，在保險契約成立時，有保險費之請求權；在承保危險事故發生時，依其承保之責任，負擔賠償之義務。         
PROMPT 甚麼是被保險人
RESPONSE 被保險人，指於保險事故發生時，遭受損害，享有賠償請求權之人；要保人亦得為被保險人。
PROMPT 甚麼是受益人
RESPONSE 受益人，指被保險人或要保人約定享有賠償請求權之人，要保人或被保險人均得為受益人。
PROMPT 甚麼是保險業
RESPONSE 保險業，指依本法組織登記，以經營保險為業之機構。


```python
from datasets import Dataset

TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{context}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer}<|eot_id|>"""

data = [
    {"text": TEMPLATE.format(context="你是保險法規助理,用中文回答問題。", question="甚麼是保險", answer="保險指當事人約定，一方交付保險費於他方，他方對於因不可預料，或不可抗力之事故所致之損害，負擔賠償財物之行為")},
    {"text": TEMPLATE.format(context="你是保險法規助理,用中文回答問題。", question="甚麼是保險人", answer="保險人指經營保險事業之各種組織，在保險契約成立時，有保險費之請求權；在承保危險事故發生時，依其承保之責任，負擔賠償之義務。")},
    {"text": TEMPLATE.format(context="你是保險法規助理,用中文回答問題。", question="甚麼是被保險人", answer="被保險人，指於保險事故發生時，遭受損害，享有賠償請求權之人；要保人亦得為被保險人。")},
    {"text": TEMPLATE.format(context="你是保險法規助理,用中文回答問題。", question="甚麼是受益人", answer="受益人，指被保險人或要保人約定享有賠償請求權之人，要保人或被保險人均得為受益人。")},
    {"text": TEMPLATE.format(context="你是保險法規助理,用中文回答問題。", question="甚麼是保險業", answer="保險業，指依本法組織登記，以經營保險為業之機構")},

]

my_dataset = Dataset.from_list(data)        

```

 In the above code block, we defined a TEMPLATE, then we created a list of dictionaries. each item in the dictionaries has a "text" key.

 Dataset class has methods for creating dataset from different sources such as CSV file, Pandas, list, json, etc.

 ** If you are familiar with pytorch's Dataset class, you can define a custom class derived from torch.util.data.Dataset class and override its __getitem__ and __len__ methods. The __getitem__ method should return a value of type dict with a "text" key.

 After creating your dataset, you can start training your model. Here is the code for this task:


```python
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig

Lora_config = LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05, bias="none")


#model = TFAutoModel.from_pretrained(pretrained_weights)
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#     model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id  
model.resize_token_embeddings(len(tokenizer))

trainer = SFTTrainer(
    peft_config=Lora_config,
    model = model,
    tokenizer = tokenizer,
    train_dataset = my_dataset,
    dataset_text_field = "text",
    max_seq_length = 256,
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 100, #100😉
        learning_rate = 2e-4,
        logging_steps = 10,
        optim = "adamw_torch",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 1234,
        output_dir = "checkpoints",
        report_to="none",
    ),
)

trainer.train()        
```

    c:\pywork\ollama\basic5_torch\prj\Lib\site-packages\huggingface_hub\utils\_deprecation.py:100: FutureWarning: Deprecated argument(s) used in '__init__': dataset_text_field, max_seq_length, dataset_num_proc. Will not be supported from version '0.13.0'.
    
    Deprecated positional argument(s) used in SFTTrainer, please use the SFTConfig to set these arguments instead.
      warnings.warn(message, FutureWarning)
    c:\pywork\ollama\basic5_torch\prj\Lib\site-packages\trl\trainer\sft_trainer.py:300: UserWarning: You passed a `max_seq_length` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`.
      warnings.warn(
    c:\pywork\ollama\basic5_torch\prj\Lib\site-packages\trl\trainer\sft_trainer.py:314: UserWarning: You passed a `dataset_num_proc` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`.
      warnings.warn(
    c:\pywork\ollama\basic5_torch\prj\Lib\site-packages\trl\trainer\sft_trainer.py:328: UserWarning: You passed a `dataset_text_field` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`.
      warnings.warn(
    


    Map (num_proc=2):   0%|          | 0/5 [00:00<?, ? examples/s]


    max_steps is given, it will override any value given in num_train_epochs
    


      0%|          | 0/100 [00:00<?, ?it/s]


    {'loss': 3.5391, 'grad_norm': 5.759950637817383, 'learning_rate': 0.00018947368421052632, 'epoch': 10.0}
    {'loss': 1.2907, 'grad_norm': 1.3779449462890625, 'learning_rate': 0.00016842105263157895, 'epoch': 20.0}
    {'loss': 0.4464, 'grad_norm': 0.5165285468101501, 'learning_rate': 0.00014736842105263158, 'epoch': 30.0}
    {'loss': 0.3223, 'grad_norm': 0.15940603613853455, 'learning_rate': 0.0001263157894736842, 'epoch': 40.0}
    {'loss': 0.305, 'grad_norm': 0.11167210340499878, 'learning_rate': 0.00010526315789473685, 'epoch': 50.0}
    {'loss': 0.2916, 'grad_norm': 0.17236745357513428, 'learning_rate': 8.421052631578948e-05, 'epoch': 60.0}
    {'loss': 0.2798, 'grad_norm': 0.13287067413330078, 'learning_rate': 6.31578947368421e-05, 'epoch': 70.0}
    {'loss': 0.2688, 'grad_norm': 0.13519032299518585, 'learning_rate': 4.210526315789474e-05, 'epoch': 80.0}
    {'loss': 0.26, 'grad_norm': 0.12910650670528412, 'learning_rate': 2.105263157894737e-05, 'epoch': 90.0}
    {'loss': 0.2545, 'grad_norm': 0.12832623720169067, 'learning_rate': 0.0, 'epoch': 100.0}
    {'train_runtime': 191.1643, 'train_samples_per_second': 16.74, 'train_steps_per_second': 0.523, 'train_loss': 0.7258142971992493, 'epoch': 100.0}
    




    TrainOutput(global_step=100, training_loss=0.7258142971992493, metrics={'train_runtime': 191.1643, 'train_samples_per_second': 16.74, 'train_steps_per_second': 0.523, 'total_flos': 2140846018560000.0, 'train_loss': 0.7258142971992493, 'epoch': 100.0})



RTX 2060 花了159.54 分鐘; 4090 3分20

注意:
1. If you get an error during training it might be because of not setting pad_token for the model and tokenizer. For solving this issue, simply add these two lines after model and tokenizer definition:
  ```python
  tokenizer.pad_token = tokenizer.eos_token
  model.config.pad_token_id = model.config.eos_token_id  
  ```
1. SFTTrainer vs Trainer
[參考](https://medium.com/@sujathamudadla1213/difference-between-trainer-class-and-sfttrainer-supervised-fine-tuning-trainer-in-hugging-face-d295344d73f7)

 after some time the training is done.

 As mentioned earlier, peft is used to decrease the GPU memory needed for fine-tuning the model. Without using peft you would need more than 16GB of memory. The GPU utilization of running the above code is shown in the following image (Tesla T4):

 To test if the model learned your data you can ask it a question from your custom dataset (instructions in section 2).

 Llama-3's impressive capabilities come with a potential trade-off. Like many large language models, it is vulnerable to catastrophic forgetting. This means that extensive fine-tuning on a specific task can cause the model to lose some of the general knowledge it acquired during its pre-training on the massive Meta dataset. To lower this risk, consider a cautious approach to fine-tuning, balancing the benefits of task-specific adaptation with the potential loss of pre-trained knowledge.


```python
prompt="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

你是保險法規助理。用中文回答問題。

<|eot_id|><|start_header_id|>user<|end_header_id|>

甚麼是受益人?

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
tokenized_chat = tokenizer.encode(prompt, return_tensors="pt")

tokenized_chat = tokenized_chat.to(device)

generate_ids = model.generate(tokenized_chat, max_new_tokens=256)


print(tokenizer.decode(generate_ids[0]))
print(tokenizer.decode(generate_ids[0],skip_special_tokens = True)) #😉skip_special_token
```

    The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    

    <|begin_of_text|><|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    你是保險法規助理。用中文回答問題。
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    
    甚麼是受益人?
    
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    受益人，指被保險人或要保人約定享有賠償請求權之人，要保人或被保險人均得為受益人。<|eot_id|>
    system
    
    你是保險法規助理。用中文回答問題。
    
    user
    
    甚麼是受益人?
    
    assistant
    
    受益人，指被保險人或要保人約定享有賠償請求權之人，要保人或被保險人均得為受益人。
    


```python
torch.save(model.state_dict(), "tmp/xxx")
```


```python
trainer.model.save_pretrained("tmp/newmodel")
```


```python
amodel = AutoModelForCausalLM.from_pretrained(
    "tmp/newmodel", 
    local_files_only=True,
    quantization_config=quantization_config, 
    device_map= device, #"cuda:0" or "auto",
    cache_dir='../../pretrain/'
    
)
 
```


    Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]



```python
#受益人，指被保險人或要保人約定享有賠償請求權之人，要保人或被保險人均得為受益人
print(tokenizer.decode(generate_ids[0],skip_special_tokens = True)) #😉skip_special_token
```

    system
    
    你是保險法規助理。用中文回答問題。
    
    user
    
    甚麼是受益人?
    
    assistant
    
    受益人，指被保險人或要保人約定享有賠償請求權之人，要保人或被保險人均得為受益人。
    


```python
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
#from trl import SFTTrainer
#from transformers import TrainingArguments
#from peft import LoraConfig
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
amodel = AutoModelForCausalLM.from_pretrained(
    "tmp/newmodel", 
    local_files_only=True,
    quantization_config=quantization_config, 
    device_map= device, #"cuda:0" or "auto",
    cache_dir='../../pretrain/'
    
) #.to(device)
tokenizer = AutoTokenizer.from_pretrained(
    "tmp/newmodel"
    #cache_dir='../../pretrain/'
)# .to(device) 
# print(tokenizer.decode(generate_ids[0],skip_special_tokens = True)) #😉skip_special_token
```


    Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[12], line 22
         13 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
         14 amodel = AutoModelForCausalLM.from_pretrained(
         15     "tmp/newmodel", 
         16     local_files_only=True,
       (...)
         20     
         21 ) #.to(device)
    ---> 22 tokenizer = AutoTokenizer.from_pretrained(
         23     "tmp/newmodel"
         24     #cache_dir='../../pretrain/'
         25 )# .to(device) 
         26 # print(tokenizer.decode(generate_ids[0],skip_special_tokens = True)) #😉skip_special_token
    

    File c:\pywork\ollama\basic5_torch\prj\Lib\site-packages\transformers\models\auto\tokenization_auto.py:877, in AutoTokenizer.from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
        875         config = AutoConfig.for_model(**config_dict)
        876     else:
    --> 877         config = AutoConfig.from_pretrained(
        878             pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        879         )
        880 config_tokenizer_class = config.tokenizer_class
        881 if hasattr(config, "auto_map") and "AutoTokenizer" in config.auto_map:
    

    File c:\pywork\ollama\basic5_torch\prj\Lib\site-packages\transformers\models\auto\configuration_auto.py:1049, in AutoConfig.from_pretrained(cls, pretrained_model_name_or_path, **kwargs)
       1046         if pattern in str(pretrained_model_name_or_path):
       1047             return CONFIG_MAPPING[pattern].from_dict(config_dict, **unused_kwargs)
    -> 1049 raise ValueError(
       1050     f"Unrecognized model in {pretrained_model_name_or_path}. "
       1051     f"Should have a `model_type` key in its {CONFIG_NAME}, or contain one of the following strings "
       1052     f"in its name: {', '.join(CONFIG_MAPPING.keys())}"
       1053 )
    

    ValueError: Unrecognized model in tmp/newmodel. Should have a `model_type` key in its config.json, or contain one of the following strings in its name: albert, align, altclip, audio-spectrogram-transformer, autoformer, bark, bart, beit, bert, bert-generation, big_bird, bigbird_pegasus, biogpt, bit, blenderbot, blenderbot-small, blip, blip-2, bloom, bridgetower, bros, camembert, canine, chameleon, chinese_clip, chinese_clip_vision_model, clap, clip, clip_text_model, clip_vision_model, clipseg, clvp, code_llama, codegen, cohere, conditional_detr, convbert, convnext, convnextv2, cpmant, ctrl, cvt, dac, data2vec-audio, data2vec-text, data2vec-vision, dbrx, deberta, deberta-v2, decision_transformer, deformable_detr, deit, depth_anything, deta, detr, dinat, dinov2, distilbert, donut-swin, dpr, dpt, efficientformer, efficientnet, electra, encodec, encoder-decoder, ernie, ernie_m, esm, falcon, falcon_mamba, fastspeech2_conformer, flaubert, flava, fnet, focalnet, fsmt, funnel, fuyu, gemma, gemma2, git, glm, glpn, gpt-sw3, gpt2, gpt_bigcode, gpt_neo, gpt_neox, gpt_neox_japanese, gptj, gptsan-japanese, granite, granitemoe, graphormer, grounding-dino, groupvit, hiera, hubert, ibert, idefics, idefics2, idefics3, imagegpt, informer, instructblip, instructblipvideo, jamba, jetmoe, jukebox, kosmos-2, layoutlm, layoutlmv2, layoutlmv3, led, levit, lilt, llama, llava, llava_next, llava_next_video, llava_onevision, longformer, longt5, luke, lxmert, m2m_100, mamba, mamba2, marian, markuplm, mask2former, maskformer, maskformer-swin, mbart, mctct, mega, megatron-bert, mgp-str, mimi, mistral, mixtral, mllama, mobilebert, mobilenet_v1, mobilenet_v2, mobilevit, mobilevitv2, moshi, mpnet, mpt, mra, mt5, musicgen, musicgen_melody, mvp, nat, nemotron, nezha, nllb-moe, nougat, nystromformer, olmo, olmoe, omdet-turbo, oneformer, open-llama, openai-gpt, opt, owlv2, owlvit, paligemma, patchtsmixer, patchtst, pegasus, pegasus_x, perceiver, persimmon, phi, phi3, phimoe, pix2struct, pixtral, plbart, poolformer, pop2piano, prophetnet, pvt, pvt_v2, qdqbert, qwen2, qwen2_audio, qwen2_audio_encoder, qwen2_moe, qwen2_vl, rag, realm, recurrent_gemma, reformer, regnet, rembert, resnet, retribert, roberta, roberta-prelayernorm, roc_bert, roformer, rt_detr, rt_detr_resnet, rwkv, sam, seamless_m4t, seamless_m4t_v2, segformer, seggpt, sew, sew-d, siglip, siglip_vision_model, speech-encoder-decoder, speech_to_text, speech_to_text_2, speecht5, splinter, squeezebert, stablelm, starcoder2, superpoint, swiftformer, swin, swin2sr, swinv2, switch_transformers, t5, table-transformer, tapas, time_series_transformer, timesformer, timm_backbone, trajectory_transformer, transfo-xl, trocr, tvlt, tvp, udop, umt5, unispeech, unispeech-sat, univnet, upernet, van, video_llava, videomae, vilt, vipllava, vision-encoder-decoder, vision-text-dual-encoder, visual_bert, vit, vit_hybrid, vit_mae, vit_msn, vitdet, vitmatte, vits, vivit, wav2vec2, wav2vec2-bert, wav2vec2-conformer, wavlm, whisper, xclip, xglm, xlm, xlm-prophetnet, xlm-roberta, xlm-roberta-xl, xlnet, xmod, yolos, yoso, zamba, zoedepth


其他參考
- [Accelerate Big Model Inference: How Does it Work?](https://www.youtube.com/watch?v=MWCSGj9jEAo&t=2s&ab_channel=HuggingFace)
- [Fine Tuning TinyLlama on Custom Dataset | Large Language Models (LLMs)](https://www.youtube.com/watch?v=3SlpXBvIqNw)

- [How large language models work, a visual intro to transformers | Chapter 5, Deep Learning](https://www.youtube.com/watch?v=wjZofJX0v4M)
- [Understanding Model Loading in Diffusers](https://medium.com/@dicksongoodluck123/understanding-model-loading-in-diffusers-db63f7ba562e)
