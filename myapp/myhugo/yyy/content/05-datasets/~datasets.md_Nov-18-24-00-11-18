---
title: datasets
description: docker log
weight: 300
---
```python
from datasets import *
```

    d:\pywork\ollama\basic5_torch\prj\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    

# datasets 基本使用

## 加载在线数据集


```python
datasets = load_dataset("madao33/new-title-chinese")
datasets
```




    DatasetDict({
        train: Dataset({
            features: ['title', 'content'],
            num_rows: 5850
        })
        validation: Dataset({
            features: ['title', 'content'],
            num_rows: 1679
        })
    })



## 加载数据集合集中的某一项任务


```python
boolq_dataset = load_dataset("super_glue", "boolq",trust_remote_code=True)
boolq_dataset
```

    Downloading data: 100%|██████████| 4.12M/4.12M [00:00<00:00, 12.6MB/s]
    Generating train split: 100%|██████████| 9427/9427 [00:00<00:00, 61328.01 examples/s]
    Generating validation split: 100%|██████████| 3270/3270 [00:00<00:00, 65644.22 examples/s]
    Generating test split: 100%|██████████| 3245/3245 [00:00<00:00, 53730.04 examples/s]
    




    DatasetDict({
        train: Dataset({
            features: ['question', 'passage', 'idx', 'label'],
            num_rows: 9427
        })
        validation: Dataset({
            features: ['question', 'passage', 'idx', 'label'],
            num_rows: 3270
        })
        test: Dataset({
            features: ['question', 'passage', 'idx', 'label'],
            num_rows: 3245
        })
    })



## 按照数据集划分进行加载


```python
dataset = load_dataset("madao33/new-title-chinese", split="train")
dataset
```




    Dataset({
        features: ['title', 'content'],
        num_rows: 5850
    })




```python
dataset = load_dataset("madao33/new-title-chinese", split="train[10:100]")
dataset
```




    Dataset({
        features: ['title', 'content'],
        num_rows: 90
    })




```python
dataset = load_dataset("madao33/new-title-chinese", split="train[:50%]")
dataset
```




    Dataset({
        features: ['title', 'content'],
        num_rows: 2925
    })




```python
dataset = load_dataset("madao33/new-title-chinese", split=["train[:50%]", "train[50%:]"])
dataset
```




    [Dataset({
         features: ['title', 'content'],
         num_rows: 2925
     }),
     Dataset({
         features: ['title', 'content'],
         num_rows: 2925
     })]



## 查看数据集


```python
datasets = load_dataset("madao33/new-title-chinese")
datasets
```




    DatasetDict({
        train: Dataset({
            features: ['title', 'content'],
            num_rows: 5850
        })
        validation: Dataset({
            features: ['title', 'content'],
            num_rows: 1679
        })
    })




```python
datasets["train"][0]
```




    {'title': '望海楼美国打“台湾牌”是危险的赌博',
     'content': '近期，美国国会众院通过法案，重申美国对台湾的承诺。对此，中国外交部发言人表示，有关法案严重违反一个中国原则和中美三个联合公报规定，粗暴干涉中国内政，中方对此坚决反对并已向美方提出严正交涉。\n事实上，中国高度关注美国国内打“台湾牌”、挑战一中原则的危险动向。近年来，作为“亲台”势力大本营的美国国会动作不断，先后通过“与台湾交往法”“亚洲再保证倡议法”等一系列“挺台”法案，“2019财年国防授权法案”也多处触及台湾问题。今年3月，美参院亲台议员再抛“台湾保证法”草案。众院议员继而在4月提出众院版的草案并在近期通过。上述法案的核心目标是强化美台关系，并将台作为美“印太战略”的重要伙伴。同时，“亲台”议员还有意制造事端。今年2月，5名共和党参议员致信众议院议长，促其邀请台湾地区领导人在国会上发表讲话。这一动议显然有悖于美国与台湾的非官方关系，其用心是实质性改变美台关系定位。\n上述动向出现并非偶然。在中美建交40周年之际，两国关系摩擦加剧，所谓“中国威胁论”再次沉渣泛起。美国对华认知出现严重偏差，对华政策中负面因素上升，保守人士甚至成立了“当前中国威胁委员会”。在此背景下，美国将台海关系作为战略抓手，通过打“台湾牌”在双边关系中增加筹码。特朗普就任后，国会对总统外交政策的约束力和塑造力加强。其实国会推动通过涉台法案对行政部门不具约束力，美政府在2018年并未提升美台官员互访级别，美军舰也没有“访问”台湾港口，保持着某种克制。但从美总统签署国会通过的法案可以看出，国会对外交产生了影响。立法也为政府对台政策提供更大空间。\n然而，美国需要认真衡量打“台湾牌”成本。首先是美国应对危机的代价。美方官员和学者已明确发出警告，美国卷入台湾问题得不偿失。美国学者曾在媒体发文指出，如果台海爆发危机，美国可能需要“援助”台湾，进而导致新的冷战乃至与中国大陆的冲突。但如果美国让台湾自己面对，则有损美国的信誉，影响美盟友对同盟关系的支持。其次是对中美关系的危害。历史证明，中美合则两利、斗则两伤。中美关系是当今世界最重要的双边关系之一，保持中美关系的稳定发展，不仅符合两国和两国人民的根本利益，也是国际社会的普遍期待。美国蓄意挑战台湾问题的底线，加剧中美关系的复杂性和不确定性，损害两国在重要领域合作，损人又害己。\n美国打“台湾牌”是一场危险的赌博。台湾问题是中国核心利益，中国政府和人民决不会对此坐视不理。中国敦促美方恪守一个中国原则和中美三个联合公报规定，阻止美国会审议推进有关法案，妥善处理涉台问题。美国悬崖勒马，才是明智之举。\n（作者系中国国际问题研究院国际战略研究所副所长）'}




```python
datasets["train"][:2]
```




    {'title': ['望海楼美国打“台湾牌”是危险的赌博', '大力推进高校治理能力建设'],
     'content': ['近期，美国国会众院通过法案，重申美国对台湾的承诺。对此，中国外交部发言人表示，有关法案严重违反一个中国原则和中美三个联合公报规定，粗暴干涉中国内政，中方对此坚决反对并已向美方提出严正交涉。\n事实上，中国高度关注美国国内打“台湾牌”、挑战一中原则的危险动向。近年来，作为“亲台”势力大本营的美国国会动作不断，先后通过“与台湾交往法”“亚洲再保证倡议法”等一系列“挺台”法案，“2019财年国防授权法案”也多处触及台湾问题。今年3月，美参院亲台议员再抛“台湾保证法”草案。众院议员继而在4月提出众院版的草案并在近期通过。上述法案的核心目标是强化美台关系，并将台作为美“印太战略”的重要伙伴。同时，“亲台”议员还有意制造事端。今年2月，5名共和党参议员致信众议院议长，促其邀请台湾地区领导人在国会上发表讲话。这一动议显然有悖于美国与台湾的非官方关系，其用心是实质性改变美台关系定位。\n上述动向出现并非偶然。在中美建交40周年之际，两国关系摩擦加剧，所谓“中国威胁论”再次沉渣泛起。美国对华认知出现严重偏差，对华政策中负面因素上升，保守人士甚至成立了“当前中国威胁委员会”。在此背景下，美国将台海关系作为战略抓手，通过打“台湾牌”在双边关系中增加筹码。特朗普就任后，国会对总统外交政策的约束力和塑造力加强。其实国会推动通过涉台法案对行政部门不具约束力，美政府在2018年并未提升美台官员互访级别，美军舰也没有“访问”台湾港口，保持着某种克制。但从美总统签署国会通过的法案可以看出，国会对外交产生了影响。立法也为政府对台政策提供更大空间。\n然而，美国需要认真衡量打“台湾牌”成本。首先是美国应对危机的代价。美方官员和学者已明确发出警告，美国卷入台湾问题得不偿失。美国学者曾在媒体发文指出，如果台海爆发危机，美国可能需要“援助”台湾，进而导致新的冷战乃至与中国大陆的冲突。但如果美国让台湾自己面对，则有损美国的信誉，影响美盟友对同盟关系的支持。其次是对中美关系的危害。历史证明，中美合则两利、斗则两伤。中美关系是当今世界最重要的双边关系之一，保持中美关系的稳定发展，不仅符合两国和两国人民的根本利益，也是国际社会的普遍期待。美国蓄意挑战台湾问题的底线，加剧中美关系的复杂性和不确定性，损害两国在重要领域合作，损人又害己。\n美国打“台湾牌”是一场危险的赌博。台湾问题是中国核心利益，中国政府和人民决不会对此坐视不理。中国敦促美方恪守一个中国原则和中美三个联合公报规定，阻止美国会审议推进有关法案，妥善处理涉台问题。美国悬崖勒马，才是明智之举。\n（作者系中国国际问题研究院国际战略研究所副所长）',
      '在推进“双一流”高校建设进程中，我们要紧紧围绕为党育人、为国育才，找准问题、破解难题，以一流意识和担当精神，大力推进高校的治理能力建设。\n增强政治引领力。坚持党对高校工作的全面领导，始终把政治建设摆在首位，增强校党委的政治领导力，全面推进党的建设各项工作。落实立德树人根本任务，把培养社会主义建设者和接班人放在中心位置。紧紧抓住思想政治工作这条生命线，全面加强师生思想政治工作，推进“三全育人”综合改革，将思想政治工作贯穿学校教育管理服务全过程，努力让学生成为德才兼备、全面发展的人才。\n提升人才聚集力。人才是创新的核心要素，创新驱动本质上是人才驱动。要坚持引育并举，建立绿色通道，探索知名专家举荐制，完善“一事一议”支持机制。在大力支持自然科学人才队伍建设的同时，实施哲学社会科学人才工程。立足实际，在条件成熟的学院探索“一院一策”改革。创新科研组织形式，为人才成长创设空间，建设更加崇尚学术、更加追求卓越、更加关爱学生、更加担当有为的学术共同体。\n培养学生竞争力。遵循学生成长成才的规律培育人才，着力培养具有国际竞争力的拔尖创新人才和各类专门人才，使优势学科、优秀教师、优质资源、优良环境围绕立德树人的根本任务配置。淘汰“水课”，打造“金课”，全力打造世界一流本科教育。深入推进研究生教育综合改革，加强事关国家重大战略的高精尖急缺人才培养，建设具有国际竞争力的研究生教育。\n激发科技创新力。在国家急需发展的领域挑大梁，就要更加聚焦科技前沿和国家需求，狠抓平台建设，包括加快牵头“武汉光源”建设步伐，积极参与国家实验室建设，建立校级大型科研仪器设备共享平台。关键核心技术领域“卡脖子”问题，归根结底是基础科学研究薄弱。要加大基础研究的支持力度，推进理论、技术和方法创新，鼓励支持重大原创和颠覆性技术创新，催生一批高水平、原创性研究成果。\n发展社会服务力。在贡献和服务中体现价值，推动合作共建、多元投入的格局，大力推进政产学研用结合，强化科技成果转移转化及产业化。探索校城融合发展、校地联动发展的新模式，深度融入地方创新发展网络，为地方经济社会发展提供人才支撑，不断拓展和优化社会服务网络。\n涵育文化软实力。加快体制机制改革，优化学校、学部、学院三级评审机制，充分发挥优秀学者特别是德才兼备的年轻学者在学术治理中的重要作用。牢固树立一流意识、紧紧围绕一流目标、认真执行一流标准，让成就一流事业成为普遍追求和行动自觉。培育具有强大凝聚力的大学文化，营造积极团结、向上向善、干事创业的氛围，让大学成为吸引和留住一大批优秀人才建功立业的沃土，让敢干事、肯干事、能干事的人有更多的荣誉感和获得感。\n建设中国特色、世界一流大学不是等得来、喊得来的，而是脚踏实地拼出来、干出来的。对标一流，深化改革，坚持按章程办学，构建以一流质量标准为核心的制度规范体系，扎实推进学校综合改革，探索更具活力、更富效率的管理体制和运行机制，我们就一定能构建起具有中国特色的现代大学治理体系，进一步提升管理服务水平和工作效能。\n（作者系武汉大学校长）']}




```python
datasets["train"]["title"][:5]
```




    ['望海楼美国打“台湾牌”是危险的赌博',
     '大力推进高校治理能力建设',
     '坚持事业为上选贤任能',
     '“大朋友”的话儿记心头',
     '用好可持续发展这把“金钥匙”']




```python
datasets["train"].column_names
```


```python
datasets["train"].features
```

## 数据集划分


```python
dataset = datasets["train"]
dataset.train_test_split(test_size=0.1)
```


```python
dataset = boolq_dataset["train"]
dataset.train_test_split(test_size=0.1, stratify_by_column="label")     # 分类数据集可以按照比例划分
```

## 数据选取与过滤


```python
# 选取
datasets["train"].select([0, 1])
```


```python
# 过滤
filter_dataset = datasets["train"].filter(lambda example: "中国" in example["title"])
```


```python
filter_dataset["title"][:5]
```

## 数据映射


```python
def add_prefix(example):
    example["title"] = 'Prefix: ' + example["title"]
    return example
```


```python
prefix_dataset = datasets.map(add_prefix)
prefix_dataset["train"][:10]["title"]
```


```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
def preprocess_function(example, tokenizer=tokenizer):
    model_inputs = tokenizer(example["content"], max_length=512, truncation=True)
    labels = tokenizer(example["title"], max_length=32, truncation=True)
    # label就是title编码的结果
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```


```python
processed_datasets = datasets.map(preprocess_function)
processed_datasets
```


```python
processed_datasets = datasets.map(preprocess_function, num_proc=4)
processed_datasets
```


```python
processed_datasets = datasets.map(preprocess_function, batched=True)
processed_datasets
```


```python
processed_datasets = datasets.map(preprocess_function, batched=True, remove_columns=datasets["train"].column_names)
processed_datasets
```

## 保存与加载


```python
processed_datasets.save_to_disk("./processed_data")
```


```python
processed_datasets = load_from_disk("./processed_data")
processed_datasets
```

# 加载本地数据集

## 直接加载文件作为数据集


```python
dataset = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train")
dataset
```


```python
dataset = Dataset.from_csv("./ChnSentiCorp_htl_all.csv")
dataset
```

## 加载文件夹内全部文件作为数据集


```python
dataset = load_dataset("csv", data_files=["./all_data/ChnSentiCorp_htl_all.csv", "./all_data/ChnSentiCorp_htl_all copy.csv"], split='train')
dataset
```

## 通过预先加载的其他格式转换加载数据集


```python
import pandas as pd

data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
data.head()
```


```python
dataset = Dataset.from_pandas(data)
dataset
```


```python
# List格式的数据需要内嵌{}，明确数据字段
data = [{"text": "abc"}, {"text": "def"}]
# data = ["abc", "def"]
Dataset.from_list(data)
```

## 通过自定义加载脚本加载数据集


```python
load_dataset("json", data_files="./cmrc2018_trial.json", field="data")
```


```python
dataset = load_dataset("./load_script.py", split="train")
dataset
```


```python
dataset[0]
```

# Dataset with DataCollator


```python
from transformers import  DataCollatorWithPadding
```


```python
dataset = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split='train')
dataset = dataset.filter(lambda x: x["review"] is not None)
dataset
```


```python
def process_function(examples):
    tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples
```


```python
tokenized_dataset = dataset.map(process_function, batched=True, remove_columns=dataset.column_names)
tokenized_dataset
```


```python
print(tokenized_dataset[:3])
```


```python
collator = DataCollatorWithPadding(tokenizer=tokenizer)
```


```python
from torch.utils.data import DataLoader
```


```python
dl = DataLoader(tokenized_dataset, batch_size=4, collate_fn=collator, shuffle=True)
```


```python
num = 0
for batch in dl:
    print(batch["input_ids"].size())
    num += 1
    if num > 10:
        break
```


```python

```
