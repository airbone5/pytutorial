---
title: evaluate
description: docker log
weight: 300
---
```python
import evaluate
```

# Evaluate使用指南

## 查看支持的评估函数


```python
# 在2024-01-11的测试中，list_evaluation_modules无法完全显示支持的评估函数，但不影响使用
# 完成的评估函数可以在 https://huggingface.co/evaluate-metric 中查看
evaluate.list_evaluation_modules()
```




    ['lvwerra/test',
     'jordyvl/ece',
     'angelina-wang/directional_bias_amplification',
     'cpllab/syntaxgym',
     'lvwerra/bary_score',
     'hack/test_metric',
     'yzha/ctc_eval',
     'codeparrot/apps_metric',
     'mfumanelli/geometric_mean',
     'daiyizheng/valid',
     'erntkn/dice_coefficient',
     'mgfrantz/roc_auc_macro',
     'Vlasta/pr_auc',
     'gorkaartola/metric_for_tp_fp_samples',
     'idsedykh/metric',
     'idsedykh/codebleu2',
     'idsedykh/codebleu',
     'idsedykh/megaglue',
     'christopher/ndcg',
     'Vertaix/vendiscore',
     'GMFTBY/dailydialogevaluate',
     'GMFTBY/dailydialog_evaluate',
     'jzm-mailchimp/joshs_second_test_metric',
     'ola13/precision_at_k',
     'yulong-me/yl_metric',
     'abidlabs/mean_iou',
     'abidlabs/mean_iou2',
     'KevinSpaghetti/accuracyk',
     'NimaBoscarino/weat',
     'ronaldahmed/nwentfaithfulness',
     'Viona/infolm',
     'kyokote/my_metric2',
     'kashif/mape',
     'Ochiroo/rouge_mn',
     'giulio98/code_eval_outputs',
     'leslyarun/fbeta_score',
     'giulio98/codebleu',
     'anz2/iliauniiccocrevaluation',
     'zbeloki/m2',
     'xu1998hz/sescore',
     'dvitel/codebleu',
     'NCSOFT/harim_plus',
     'JP-SystemsX/nDCG',
     'sportlosos/sescore',
     'Drunper/metrica_tesi',
     'jpxkqx/peak_signal_to_noise_ratio',
     'jpxkqx/signal_to_reconstruction_error',
     'hpi-dhc/FairEval',
     'lvwerra/accuracy_score',
     'ybelkada/cocoevaluate',
     'harshhpareek/bertscore',
     'posicube/mean_reciprocal_rank',
     'bstrai/classification_report',
     'omidf/squad_precision_recall',
     'Josh98/nl2bash_m',
     'BucketHeadP65/confusion_matrix',
     'BucketHeadP65/roc_curve',
     'yonting/average_precision_score',
     'transZ/test_parascore',
     'transZ/sbert_cosine',
     'hynky/sklearn_proxy',
     'xu1998hz/sescore_english_mt',
     'xu1998hz/sescore_german_mt',
     'xu1998hz/sescore_english_coco',
     'xu1998hz/sescore_english_webnlg',
     'unnati/kendall_tau_distance',
     'Viona/fuzzy_reordering',
     'Viona/kendall_tau',
     'lhy/hamming_loss',
     'lhy/ranking_loss',
     'Muennighoff/code_eval_octopack',
     'yuyijiong/quad_match_score',
     'Splend1dchan/cosine_similarity',
     'AlhitawiMohammed22/CER_Hu-Evaluation-Metrics',
     'Yeshwant123/mcc',
     'transformersegmentation/segmentation_scores',
     'sma2023/wil',
     'chanelcolgate/average_precision',
     'ckb/unigram',
     'Felipehonorato/eer',
     'manueldeprada/beer',
     'shunzh/apps_metric',
     'He-Xingwei/sari_metric',
     'langdonholmes/cohen_weighted_kappa',
     'fschlatt/ner_eval',
     'hyperml/balanced_accuracy',
     'brian920128/doc_retrieve_metrics',
     'guydav/restrictedpython_code_eval',
     'k4black/codebleu',
     'Natooz/ece',
     'ingyu/klue_mrc',
     'Vipitis/shadermatch',
     'gabeorlanski/bc_eval',
     'jjkim0807/code_eval',
     'vichyt/metric-codebleu',
     'repllabs/mean_reciprocal_rank',
     'repllabs/mean_average_precision',
     'mtc/fragments',
     'DarrenChensformer/eval_keyphrase',
     'kedudzic/charmatch',
     'Vallp/ter',
     'DarrenChensformer/relation_extraction',
     'Ikala-allen/relation_extraction',
     'danieldux/hierarchical_softmax_loss',
     'nlpln/tst',
     'bdsaglam/jer',
     'davebulaval/meaningbert',
     'fnvls/bleu1234',
     'fnvls/bleu_1234',
     'nevikw39/specificity',
     'yqsong/execution_accuracy',
     'shalakasatheesh/squad_v2',
     'arthurvqin/pr_auc',
     'd-matrix/dmx_perplexity',
     'akki2825/accents_unplugged_eval',
     'juliakaczor/accents_unplugged_eval',
     'chimene/accents_unplugged_eval',
     'Vickyage/accents_unplugged_eval',
     'Qui-nn/accents_unplugged_eval',
     'TelEl/accents_unplugged_eval',
     'livvie/accents_unplugged_eval',
     'DaliaCaRo/accents_unplugged_eval',
     'alvinasvk/accents_unplugged_eval',
     'LottieW/accents_unplugged_eval',
     'LuckiestOne/valid_efficiency_score',
     'Fritz02/execution_accuracy',
     'huanghuayu/multiclass_brier_score',
     'jialinsong/apps_metric',
     'DoctorSlimm/bangalore_score',
     'agkphysics/ccc',
     'DoctorSlimm/kaushiks_criteria',
     'CZLC/rouge_raw',
     'bascobasculino/mot-metrics',
     'SEA-AI/mot-metrics',
     'SEA-AI/det-metrics',
     'saicharan2804/my_metric',
     'red1bluelost/evaluate_genericify_cpp',
     'maksymdolgikh/seqeval_with_fbeta',
     'Bekhouche/NED',
     'danieldux/isco_hierarchical_accuracy',
     'ginic/phone_errors',
     'haotongye-shopee/ppl',
     'berkatil/map',
     'DarrenChensformer/action_generation',
     'buelfhood/fbeta_score',
     'danasone/ru_errant',
     'helena-balabin/youden_index',
     'SEA-AI/panoptic-quality',
     'SEA-AI/box-metrics',
     'MathewShen/bleu',
     'berkatil/mrr',
     'nbansal/semf1',
     'SEA-AI/horizon-metrics',
     'bdsaglam/musique',
     'maysonma/lingo_judge_metric',
     'dannashao/span_metric',
     'Aye10032/loss_metric',
     'ag2435/my_metric',
     'mlcore/arxiv_score',
     'jiiiiin/code_eval_octopack',
     'ncoop57/levenshtein_distance',
     'kaleidophon/almost_stochastic_order',
     'lvwerra/element_count',
     'prb977/cooccurrence_count',
     'NimaBoscarino/pseudo_perplexity',
     'ybelkada/toxicity',
     'ronaldahmed/ccl_win',
     'christopher/tokens_per_byte',
     'lsy641/distinct',
     'grepLeigh/perplexity',
     'Charles95/element_count',
     'Charles95/accuracy']




```python
evaluate.list_evaluation_modules(
  module_type="comparison",
  include_community=False,
  with_details=True)
```




    []



## 加载评估函数


```python
accuracy = evaluate.load("accuracy")
```


    Downloading builder script:   0%|          | 0.00/1.67k [00:00<?, ?B/s]



```python
accuracy
```




    EvaluationModule(name: "accuracy", module_type: "metric", features: {'predictions': Value(dtype='int32', id=None), 'references': Value(dtype='int32', id=None)}, usage: """
    Args:
        predictions (`list` of `int`): Predicted labels.
        references (`list` of `int`): Ground truth labels.
        normalize (`boolean`): If set to False, returns the number of correctly classified samples. Otherwise, returns the fraction of correctly classified samples. Defaults to True.
        sample_weight (`list` of `float`): Sample weights Defaults to None.
    
    Returns:
        accuracy (`float` or `int`): Accuracy score. Minimum possible value is 0. Maximum possible value is 1.0, or the number of examples input, if `normalize` is set to `True`.. A higher score means higher accuracy.
    
    Examples:
    
        Example 1-A simple example
            >>> accuracy_metric = evaluate.load("accuracy")
            >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])
            >>> print(results)
            {'accuracy': 0.5}
    
        Example 2-The same as Example 1, except with `normalize` set to `False`.
            >>> accuracy_metric = evaluate.load("accuracy")
            >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], normalize=False)
            >>> print(results)
            {'accuracy': 3.0}
    
        Example 3-The same as Example 1, except with `sample_weight` set.
            >>> accuracy_metric = evaluate.load("accuracy")
            >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], sample_weight=[0.5, 2, 0.7, 0.5, 9, 0.4])
            >>> print(results)
            {'accuracy': 0.8778625954198473}
    """, stored examples: 0)



## 查看函数说明


```python
print(accuracy.description)
```

    
    Accuracy is the proportion of correct predictions among the total number of cases processed. It can be computed with:
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
     Where:
    TP: True positive
    TN: True negative
    FP: False positive
    FN: False negative
    
    


```python
print(accuracy.inputs_description)
```

    
    Args:
        predictions (`list` of `int`): Predicted labels.
        references (`list` of `int`): Ground truth labels.
        normalize (`boolean`): If set to False, returns the number of correctly classified samples. Otherwise, returns the fraction of correctly classified samples. Defaults to True.
        sample_weight (`list` of `float`): Sample weights Defaults to None.
    
    Returns:
        accuracy (`float` or `int`): Accuracy score. Minimum possible value is 0. Maximum possible value is 1.0, or the number of examples input, if `normalize` is set to `True`.. A higher score means higher accuracy.
    
    Examples:
    
        Example 1-A simple example
            >>> accuracy_metric = evaluate.load("accuracy")
            >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])
            >>> print(results)
            {'accuracy': 0.5}
    
        Example 2-The same as Example 1, except with `normalize` set to `False`.
            >>> accuracy_metric = evaluate.load("accuracy")
            >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], normalize=False)
            >>> print(results)
            {'accuracy': 3.0}
    
        Example 3-The same as Example 1, except with `sample_weight` set.
            >>> accuracy_metric = evaluate.load("accuracy")
            >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], sample_weight=[0.5, 2, 0.7, 0.5, 9, 0.4])
            >>> print(results)
            {'accuracy': 0.8778625954198473}
    
    


```python
accuracy
```




    EvaluationModule(name: "accuracy", module_type: "metric", features: {'predictions': Value(dtype='int32', id=None), 'references': Value(dtype='int32', id=None)}, usage: """
    Args:
        predictions (`list` of `int`): Predicted labels.
        references (`list` of `int`): Ground truth labels.
        normalize (`boolean`): If set to False, returns the number of correctly classified samples. Otherwise, returns the fraction of correctly classified samples. Defaults to True.
        sample_weight (`list` of `float`): Sample weights Defaults to None.
    
    Returns:
        accuracy (`float` or `int`): Accuracy score. Minimum possible value is 0. Maximum possible value is 1.0, or the number of examples input, if `normalize` is set to `True`.. A higher score means higher accuracy.
    
    Examples:
    
        Example 1-A simple example
            >>> accuracy_metric = evaluate.load("accuracy")
            >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])
            >>> print(results)
            {'accuracy': 0.5}
    
        Example 2-The same as Example 1, except with `normalize` set to `False`.
            >>> accuracy_metric = evaluate.load("accuracy")
            >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], normalize=False)
            >>> print(results)
            {'accuracy': 3.0}
    
        Example 3-The same as Example 1, except with `sample_weight` set.
            >>> accuracy_metric = evaluate.load("accuracy")
            >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], sample_weight=[0.5, 2, 0.7, 0.5, 9, 0.4])
            >>> print(results)
            {'accuracy': 0.8778625954198473}
    """, stored examples: 0)



## 评估指标计算——全局计算


```python
accuracy = evaluate.load("accuracy")
results = accuracy.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])
results
```




    {'accuracy': 0.5}



## 评估指标计算——迭代计算


```python
accuracy = evaluate.load("accuracy")
for ref, pred in zip([0,1,0,1], [1,0,0,1]):
    accuracy.add(references=ref, predictions=pred)
accuracy.compute()
```




    {'accuracy': 0.5}




```python
accuracy = evaluate.load("accuracy")
for refs, preds in zip([[0,1],[0,1]], [[1,0],[0,1]]):
    accuracy.add_batch(references=refs, predictions=preds)
accuracy.compute()
```




    {'accuracy': 0.5}



## 多个评估指标计算


```python
clf_metrics = evaluate.combine(["accuracy", "f1", "recall", "precision"])
clf_metrics
```




    <evaluate.module.CombinedEvaluations at 0x19dd0e2e100>




```python
clf_metrics
```




    <evaluate.module.CombinedEvaluations at 0x19dd0e2e100>




```python
clf_metrics.compute(predictions=[0, 1, 0], references=[0, 1, 1])
```




    {'accuracy': 0.6666666666666666,
     'f1': 0.6666666666666666,
     'recall': 0.5,
     'precision': 1.0}



## 评估结果对比可视化


```python
from evaluate.visualization import radar_plot   # 目前只支持雷达图
```


```python
data = [
   {"accuracy": 0.99, "precision": 0.8, "f1": 0.95, "latency_in_seconds": 33.6},
   {"accuracy": 0.98, "precision": 0.87, "f1": 0.91, "latency_in_seconds": 11.2},
   {"accuracy": 0.98, "precision": 0.78, "f1": 0.88, "latency_in_seconds": 87.6}, 
   {"accuracy": 0.88, "precision": 0.78, "f1": 0.81, "latency_in_seconds": 101.6}
   ]
model_names = ["Model 1", "Model 2", "Model 3", "Model 4"]
```


```python
plot = radar_plot(data=data, model_names=model_names)
```


    
![png](output_24_0.png)
    



```python

```
