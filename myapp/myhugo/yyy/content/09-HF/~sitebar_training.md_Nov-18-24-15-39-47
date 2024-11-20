---
title: sitebar_training
description: docker log
weight: 300
---
# 資料集存檔

## import pandas as pd
## import datasets
```python
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
 
dataset = load_dataset('csv', data_files={'train': 'train_spam.csv', 'test': 'test_spam.csv'})
 
dataset
```
DatasetDict({
    train: Dataset({
        features: ['text', 'target'],
        num_rows: 3900
    })
    test: Dataset({
        features: ['text', 'target'],
        num_rows: 1672
    })
})


In order to save the dataset, we have the following options:
```python 
# Arrow format
dataset.save_to_disk()
 
# CSV format
dataset.to_csv()
 
# JSON format
dataset.to_json()
 
# Parquet
dataset.to_parquet()
```
Let’s choose the arrow format and save the dataset to the disk.

```
dataset.save_to_disk('ham_spam_dataset')
```
Now, we are ready to load the data from the disk.

```
dataset = load_from_disk('ham_spam_dataset')
dataset
```
DatasetDict({
    train: Dataset({
        features: ['text', 'target'],
        num_rows: 3900
    })
    test: Dataset({
        features: ['text', 'target'],
        num_rows: 1672
    })
})

Save a Dataset to CSV format
A Dataset is a dictionary with 1 or more Datasets. In order to save each dataset into a different CSV file we will need to iterate over the dataset. For example:
```
from datasets import loda_dataset
 
# assume that we have already loaded the dataset called "dataset"
for split, data in dataset.items():
    data.to_csv(f"my-dataset-{split}.csv", index = None)
```    

[存模型](https://stackoverflow.com/questions/42703500/how-do-i-save-a-trained-model-in-pytorch)
