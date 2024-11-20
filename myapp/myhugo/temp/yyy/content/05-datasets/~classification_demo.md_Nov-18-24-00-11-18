---
title: classification_demo
description: docker log
weight: 300
---
# 文本分类实例

## Step1 导入相关包


```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
```

## Step2 加载数据集
[相關參考](https://ithelp.ithome.com.tw/m/articles/10296418)


```python
dataset = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train")
dataset = dataset.filter(lambda x: x["review"] is not None)
dataset
```

## Step4 划分数据集


```python
datasets = dataset.train_test_split(test_size=0.1)
datasets
```

## Step5 创建Dataloader


```python
import torch

tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")

def process_function(examples):
    tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples

tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)
tokenized_datasets
```


```python
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

trainset, validset = tokenized_datasets["train"], tokenized_datasets["test"]
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer))
validloader = DataLoader(validset, batch_size=64, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer))
```


```python
next(enumerate(validloader))[1]
```

## Step6 创建模型及优化器


```python
from torch.optim import Adam

model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")

if torch.cuda.is_available():
    model = model.cuda()
```


```python
optimizer = Adam(model.parameters(), lr=2e-5)
```

## Step7 训练与验证


```python
def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    return acc_num / len(validset)

def train(epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                print(f"ep: {ep}, global_step: {global_step}, loss: {output.loss.item()}")
            global_step += 1
        acc = evaluate()
        print(f"ep: {ep}, acc: {acc}")
```

## Step8 模型训练


```python
train()
```

## Step9 模型预测


```python
sen = "我觉得这家酒店不错，饭很好吃！"
id2_label = {0: "差评！", 1: "好评！"}
model.eval()
with torch.inference_mode():
    inputs = tokenizer(sen, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1)
    print(f"输入：{sen}\n模型预测结果:{id2_label.get(pred.item())}")
```


```python
from transformers import pipeline

model.config.id2label = id2_label
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
```


```python
pipe(sen)
```


```python

```
