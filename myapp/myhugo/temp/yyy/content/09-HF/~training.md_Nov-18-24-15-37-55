---
title: training
description: docker log
weight: 300
---
```python
# Transformers installation
! pip install transformers datasets
# To install from source instead of the last release, comment the command above and uncomment the following one.
# ! pip install git+https://github.com/huggingface/transformers.git
```

# Fine-tune a pretrained model

There are significant benefits to using a pretrained model. It reduces computation costs, your carbon footprint, and allows you to use state-of-the-art models without having to train one from scratch. 🤗 Transformers provides access to thousands of pretrained models for a wide range of tasks. When you use a pretrained model, you train it on a dataset specific to your task. This is known as fine-tuning, an incredibly powerful training technique. In this tutorial, you will fine-tune a pretrained model with a deep learning framework of your choice:

* Fine-tune a pretrained model with 🤗 Transformers [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer).
* Fine-tune a pretrained model in TensorFlow with Keras.
* Fine-tune a pretrained model in native PyTorch.

<a id='data-processing'></a>

## Prepare a dataset


```python
#@title
from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/_BZearw7f0w?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')
```




<iframe width="560" height="315" src="https://www.youtube.com/embed/_BZearw7f0w?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>



Before you can fine-tune a pretrained model, download a dataset and prepare it for training. The previous tutorial showed you how to process data for training, and now you get an opportunity to put those skills to the test!

Begin by loading the [Yelp Reviews](https://huggingface.co/datasets/yelp_review_full) dataset:


```python
from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
dataset["train"][100]
```


    README.md:   0%|          | 0.00/6.72k [00:00<?, ?B/s]


    c:\pywork\ollama\basic5_torch\prj\Lib\site-packages\huggingface_hub\file_download.py:139: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\linchao\.cache\huggingface\hub\datasets--yelp_review_full. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
    To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
      warnings.warn(message)
    


    train-00000-of-00001.parquet:   0%|          | 0.00/299M [00:00<?, ?B/s]



    test-00000-of-00001.parquet:   0%|          | 0.00/23.5M [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/650000 [00:00<?, ? examples/s]



    Generating test split:   0%|          | 0/50000 [00:00<?, ? examples/s]





    {'label': 0,
     'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...that takes something special!\\nThe cashier took my friends\'s order, then promptly ignored me. I had to force myself in front of a cashier who opened his register to wait on the person BEHIND me. I waited over five minutes for a gigantic order that included precisely one kid\'s meal. After watching two people who ordered after me be handed their food, I asked where mine was. The manager started yelling at the cashiers for \\"serving off their orders\\" when they didn\'t have their food. But neither cashier was anywhere near those controls, and the manager was the one serving food to customers and clearing the boards.\\nThe manager was rude when giving me my order. She didn\'t make sure that I had everything ON MY RECEIPT, and never even had the decency to apologize that I felt I was getting poor service.\\nI\'ve eaten at various McDonalds restaurants for over 30 years. I\'ve worked at more than one location. I expect bad days, bad moods, and the occasional mistake. But I have yet to have a decent experience at this store. It will remain a place I avoid unless someone in my party needs to avoid illness from low blood sugar. Perhaps I should go back to the racially biased service of Steak n Shake instead!'}



As you now know, you need a tokenizer to process the text and include a padding and truncation strategy to handle any variable sequence lengths. To process your dataset in one step, use 🤗 Datasets [`map`](https://huggingface.co/docs/datasets/process.html#map) method to apply a preprocessing function over the entire dataset:


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

If you like, you can create a smaller subset of the full dataset to fine-tune on to reduce the time it takes:


```python
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

<a id='trainer'></a>

## Train

At this point, you should follow the section corresponding to the framework you want to use. You can use the links
in the right sidebar to jump to the one you want - and if you want to hide all of the content for a given framework,
just use the button at the top-right of that framework's block!


```python
#@title
from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/nvBXf7s7vTI?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')
```




<iframe width="560" height="315" src="https://www.youtube.com/embed/nvBXf7s7vTI?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>



## Train with PyTorch Trainer

🤗 Transformers provides a [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) class optimized for training 🤗 Transformers models, making it easier to start training without manually writing your own training loop. The [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) API supports a wide range of training options and features such as logging, gradient accumulation, and mixed precision.

Start by loading your model and specify the number of expected labels. From the Yelp Review [dataset card](https://huggingface.co/datasets/yelp_review_full#data-fields), you know there are five labels:


```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
```

<Tip>

You will see a warning about some of the pretrained weights not being used and some weights being randomly
initialized. Don't worry, this is completely normal! The pretrained head of the BERT model is discarded, and replaced with a randomly initialized classification head. You will fine-tune this new model head on your sequence classification task, transferring the knowledge of the pretrained model to it.

</Tip>

### Training hyperparameters

Next, create a [TrainingArguments](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) class which contains all the hyperparameters you can tune as well as flags for activating different training options. For this tutorial you can start with the default training [hyperparameters](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments), but feel free to experiment with these to find your optimal settings.

Specify where to save the checkpoints from your training:


```python
from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")
```

### Evaluate

[Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) does not automatically evaluate model performance during training. You'll need to pass [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) a function to compute and report metrics. The [🤗 Evaluate](https://huggingface.co/docs/evaluate/index) library provides a simple [`accuracy`](https://huggingface.co/spaces/evaluate-metric/accuracy) function you can load with the [evaluate.load](https://huggingface.co/docs/evaluate/main/en/package_reference/loading_methods#evaluate.load) (see this [quicktour](https://huggingface.co/docs/evaluate/a_quick_tour) for more information) function:


```python
import numpy as np
import evaluate

metric = evaluate.load("accuracy")
```

Call `compute` on `metric` to calculate the accuracy of your predictions. Before passing your predictions to `compute`, you need to convert the predictions to logits (remember all 🤗 Transformers models return logits):


```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

If you'd like to monitor your evaluation metrics during fine-tuning, specify the `evaluation_strategy` parameter in your training arguments to report the evaluation metric at the end of each epoch:


```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
```

### Trainer

Create a [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) object with your model, training arguments, training and test datasets, and evaluation function:


```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
```

Then fine-tune your model by calling [train()](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer.train):


```python
trainer.train()
```

<a id='pytorch_native'></a>

## Train in native PyTorch


```python
#@title
from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/Dh9CL8fyG80?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')
```




<iframe width="560" height="315" src="https://www.youtube.com/embed/Dh9CL8fyG80?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>



[Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) takes care of the training loop and allows you to fine-tune a model in a single line of code. For users who prefer to write their own training loop, you can also fine-tune a 🤗 Transformers model in native PyTorch.

At this point, you may need to restart your notebook or execute the following code to free some memory:


```python
del model
del trainer
torch.cuda.empty_cache()
```

Next, manually postprocess `tokenized_dataset` to prepare it for training.

1. Remove the `text` column because the model does not accept raw text as an input:

    ```py
    >>> tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    ```

2. Rename the `label` column to `labels` because the model expects the argument to be named `labels`:

    ```py
    >>> tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    ```

3. Set the format of the dataset to return PyTorch tensors instead of lists:

    ```py
    >>> tokenized_datasets.set_format("torch")
    ```

Then create a smaller subset of the dataset as previously shown to speed up the fine-tuning:


```python
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

### DataLoader

Create a `DataLoader` for your training and test datasets so you can iterate over batches of data:


```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
```

Load your model with the number of expected labels:


```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
```

### Optimizer and learning rate scheduler

Create an optimizer and learning rate scheduler to fine-tune the model. Let's use the [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) optimizer from PyTorch:


```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
```

Create the default learning rate scheduler from [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer):


```python
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
```

Lastly, specify `device` to use a GPU if you have access to one. Otherwise, training on a CPU may take several hours instead of a couple of minutes.


```python
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
```

<Tip>

Get free access to a cloud GPU if you don't have one with a hosted notebook like [Colaboratory](https://colab.research.google.com/) or [SageMaker StudioLab](https://studiolab.sagemaker.aws/).

</Tip>

Great, now you are ready to train! 🥳

### Training loop

To keep track of your training progress, use the [tqdm](https://tqdm.github.io/) library to add a progress bar over the number of training steps:


```python
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

### Evaluate

Just like how you added an evaluation function to [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer), you need to do the same when you write your own training loop. But instead of calculating and reporting the metric at the end of each epoch, this time you'll accumulate all the batches with `add_batch` and calculate the metric at the very end.


```python
import evaluate

metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```

<a id='additional-resources'></a>

## Additional resources

For more fine-tuning examples, refer to:

- [🤗 Transformers Examples](https://github.com/huggingface/transformers/tree/main/examples) includes scripts
  to train common NLP tasks in PyTorch and TensorFlow.

- [🤗 Transformers Notebooks](https://huggingface.co/docs/transformers/main/en/notebooks) contains various notebooks on how to fine-tune a model for specific tasks in PyTorch and TensorFlow.
