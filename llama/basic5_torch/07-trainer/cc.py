     
from transformers import AutoTokenizer , AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import torch
    
if __name__ == '__main__':
    # %% [markdown]
    # ## Step2 加載數據集

    # %%
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train")
    dataset = dataset.filter(lambda x: x["review"] is not None )
    

    datasets = dataset.train_test_split(test_size=0.1)
    

    # %% [markdown]
    # ## Step4 數據集預處理

    # %%

    tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")

    def process_function(examples):
        tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True) # tokenized_examples 裡面有 input_ids', 'attention_mask' 等等        
        tokenized_examples["labels"] = examples["label"] 
        return tokenized_examples

    tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)
    
    model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")

    

    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

     
    def eval_metric(eval_predict):
        predictions, labels = eval_predict
        predictions = predictions.argmax(axis=-1)
        acc = acc_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels)
        acc.update(f1)
        return acc

     
    train_args = TrainingArguments(output_dir="./checkpoints",      # 輸出文件夾
                                   per_device_train_batch_size=64,  # 訓練時的batch_size
                                   per_device_eval_batch_size=128,  # 驗證時的batch_size
                                   logging_steps=10,                # log 打印的頻率
                                   evaluation_strategy="epoch",     # 評估策略
                                   save_strategy="epoch",           # 保存策略
                                   save_total_limit=3,              # 最大保存數
                                   learning_rate=2e-5,              # 學習率
                                   weight_decay=0.01,               # weight_decay
                                   metric_for_best_model="f1",      # 設定評估指標
                                   load_best_model_at_end=True)     # 訓練完成後加載最優模型
   
     
    from transformers import DataCollatorWithPadding
    trainer = Trainer(model=model, 
                      args=train_args, 
                      train_dataset=tokenized_datasets["train"] , 
                      eval_dataset=tokenized_datasets["test"] , 
                      data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                      compute_metrics=eval_metric)

     
    trainer.train()

    exit()
    #12m 20.7s

    # %% [markdown]
    # ## Step10 模型評估

    # %%
    trainer.evaluate(tokenized_datasets["test"])

    # %% [markdown]
    # ## Step11 模型預測

    # %%
    trainer.predict(tokenized_datasets["test"])

    # %%
    from transformers import pipeline

    id2_label = id2_label = {0: "差評！", 1: "好評！"}
    model.config.id2label = id2_label
    #pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device="cpu")
    #❌pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # %%
    sen = "我覺得不錯！"
    pipe(sen)

    # %%



