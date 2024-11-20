---
title: classification_demo
description: docker log
weight: 300
---
# 文本分类实例

## Step1 导入相关包


```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
```

## Step2 加载数据集


```python
dataset = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train")
dataset = dataset.filter(lambda x: x["review"] is not None)
dataset
```




    Dataset({
        features: ['label', 'review'],
        num_rows: 7765
    })



## Step3 划分数据集


```python
datasets = dataset.train_test_split(test_size=0.1)
datasets
```

## Step4 数据集预处理


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

## Step5 创建模型


```python
model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
```


```python
model.config
```

## Step6 创建评估函数


```python
import evaluate

acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
```


```python
def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc
```

## Step7 创建TrainingArguments


```python
train_args = TrainingArguments(output_dir="./checkpoints",      # 输出文件夹
                               per_device_train_batch_size=64,  # 训练时的batch_size
                               per_device_eval_batch_size=128,  # 验证时的batch_size
                               logging_steps=10,                # log 打印的频率
                               evaluation_strategy="epoch",     # 评估策略
                               save_strategy="epoch",           # 保存策略
                               save_total_limit=3,              # 最大保存数
                               learning_rate=2e-5,              # 学习率
                               weight_decay=0.01,               # weight_decay
                               metric_for_best_model="f1",      # 设定评估指标
                               load_best_model_at_end=True)     # 训练完成后加载最优模型
train_args
```

## Step8 创建Trainer


```python
from transformers import DataCollatorWithPadding
trainer = Trainer(model=model, 
                  args=train_args, 
                  train_dataset=tokenized_datasets["train"], 
                  eval_dataset=tokenized_datasets["test"], 
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_metric)
```

## Step9 模型训练


```python
trainer.train()
```

## Step10 模型评估


```python
trainer.evaluate(tokenized_datasets["test"])
```

## Step11 模型预测


```python
trainer.predict(tokenized_datasets["test"])
```


```python
from transformers import pipeline

id2_label = id2_label = {0: "差评！", 1: "好评！"}
model.config.id2label = id2_label
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
```


```python
sen = "我觉得不错！"
pipe(sen)
```


```python

```
