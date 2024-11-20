---
title: sidebar1
description: docker log
weight: 300
---
相關參考
- [hugginface CSV](https://stackoverflow.com/questions/77020278/how-to-load-a-huggingface-dataset-from-local-path)
- [official datasets introduction](https://www.kaggle.com/code/nbroad/intro-to-hugging-face-datasets)

torch, pytorch
HuggingFace的datasets库中load_dataset方法使用


datasets是抱抱脸开发的一个数据集python库，可以很方便的从Hugging Face Hub里下载数据，也可很方便的从本地加载数据集，本文主要对load_dataset方法的使用进行详细说明

## load_dataset参数
load_dataset有以下参数，具体可参考  

```python
def load_dataset(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Union[Dict, List] = None,
    split: Optional[Union[str, Split]] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[GenerateMode] = None,
    ignore_verifications: bool = False,
    save_infos: bool = False,
    script_version: Optional[Union[str, Version]] = None,
    **config_kwargs,
) -> Union[DatasetDict, Dataset]:
``` 
- path：参数path表示数据集的名字或者路径。可以是如下几种形式（每种形式的使用方式后面会详细说明）
  - 数据集的名字，比如imdb、glue
  - 数据集文件格式，比如json、csv、parquet、txt
  - 数据集目录中的处理数据集的脚本（.py)文件，比如“glue/glue.py”
- name：参数name表示数据集中的子数据集，当一个数据集包含多个数据集时，就需要这个参数，比如- glue数据集下就包含"sst2"、“cola”、"qqp"等多个子数据集，此时就需要指定name来表示加载哪一个子数据集
- data_dir：数据集所在的目录
- data_files：数据集文件
- cache_dir：构建的数据集缓存目录，方便下次快速加载
以上为一些常用且比较重要的参数，其他参数很少用到因此在此处不再详细说明，下面会通过一些case更加具体的说明各种用法

#### 详细用法
从HuggingFace Hub上加载数据

首先我们可以通过如下方式查看Hubs上有哪些数据集
```python
from datasets import list_datasets

datasets_list = list_datasets()
print( len(datasets_list))
print(datasets_list[:10])
``` 

输出如下

```txt
['acronym_identification', 'ade_corpus_v2', 'adversarial_qa', 'aeslc', 'afrikaans_ner_corpus', 'ag_news', 'ai2_arc', 'air_dialogue', 'ajgt_twitter_ar', 'allegro_reviews']
```

后面通过直接指定path等于相关数据集的名字就能下载并加载相关数据集
```python
from datasets import load_dataset
dataset = load_dataset(path='squad', split='train')
``` 

### 測試1
從CSV中載入,第一ROW有欄位名稱,但是split=true沒用


```python
from datasets import *
dataset = load_dataset("csv",data_files="../../../python/dataset/iris2.csv", split="train")
dataset
```




    Dataset({
        features: ['f1', 'f2', 'f3', 'f4', 'f5'],
        num_rows: 150
    })




```python
from datasets import *
ds = load_dataset("csv",data_files="../../../python/dataset/iris2.csv")
ds['train'] ## 和ds結果一樣
```




    Dataset({
        features: ['f1', 'f2', 'f3', 'f4', 'f5'],
        num_rows: 150
    })




```python
# 150*0.2=30
train_testvalid = ds['train'].train_test_split(test_size=0.2)
train_testvalid

```




    DatasetDict({
        train: Dataset({
            features: ['f1', 'f2', 'f3', 'f4', 'f5'],
            num_rows: 120
        })
        test: Dataset({
            features: ['f1', 'f2', 'f3', 'f4', 'f5'],
            num_rows: 30
        })
    })




```python
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
# gather everyone if you want to have a single DatasetDict
ds = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})
ds
```
