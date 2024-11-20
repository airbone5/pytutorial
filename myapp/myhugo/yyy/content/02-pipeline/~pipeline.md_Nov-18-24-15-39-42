---
title: pipeline
description: docker log
weight: 300
---
 # 查看Pipeline支持的任務類型


```python
from transformers.pipelines import SUPPORTED_TASKS
for item in SUPPORTED_TASKS.keys():
  print(item)

```

    audio-classification
    automatic-speech-recognition
    text-to-audio
    feature-extraction
    text-classification
    token-classification
    question-answering
    table-question-answering
    visual-question-answering
    document-question-answering
    fill-mask
    summarization
    translation
    text2text-generation
    text-generation
    zero-shot-classification
    zero-shot-image-classification
    zero-shot-audio-classification
    image-classification
    image-feature-extraction
    image-segmentation
    image-to-text
    object-detection
    zero-shot-object-detection
    depth-estimation
    video-classification
    mask-generation
    image-to-image
    


```python
# %%
for k, v in SUPPORTED_TASKS.items():
    print(k, v)

```

    audio-classification {'impl': <class 'transformers.pipelines.audio_classification.AudioClassificationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForAudioClassification'>,), 'default': {'model': {'pt': ('superb/wav2vec2-base-superb-ks', '372e048')}}, 'type': 'audio'}
    automatic-speech-recognition {'impl': <class 'transformers.pipelines.automatic_speech_recognition.AutomaticSpeechRecognitionPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForCTC'>, <class 'transformers.models.auto.modeling_auto.AutoModelForSpeechSeq2Seq'>), 'default': {'model': {'pt': ('facebook/wav2vec2-base-960h', '22aad52')}}, 'type': 'multimodal'}
    text-to-audio {'impl': <class 'transformers.pipelines.text_to_audio.TextToAudioPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForTextToWaveform'>, <class 'transformers.models.auto.modeling_auto.AutoModelForTextToSpectrogram'>), 'default': {'model': {'pt': ('suno/bark-small', '1dbd7a1')}}, 'type': 'text'}
    feature-extraction {'impl': <class 'transformers.pipelines.feature_extraction.FeatureExtractionPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModel'>,), 'default': {'model': {'pt': ('distilbert/distilbert-base-cased', '6ea8117'), 'tf': ('distilbert/distilbert-base-cased', '6ea8117')}}, 'type': 'multimodal'}
    text-classification {'impl': <class 'transformers.pipelines.text_classification.TextClassificationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForSequenceClassification'>,), 'default': {'model': {'pt': ('distilbert/distilbert-base-uncased-finetuned-sst-2-english', '714eb0f'), 'tf': ('distilbert/distilbert-base-uncased-finetuned-sst-2-english', '714eb0f')}}, 'type': 'text'}
    token-classification {'impl': <class 'transformers.pipelines.token_classification.TokenClassificationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForTokenClassification'>,), 'default': {'model': {'pt': ('dbmdz/bert-large-cased-finetuned-conll03-english', '4c53496'), 'tf': ('dbmdz/bert-large-cased-finetuned-conll03-english', '4c53496')}}, 'type': 'text'}
    question-answering {'impl': <class 'transformers.pipelines.question_answering.QuestionAnsweringPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForQuestionAnswering'>,), 'default': {'model': {'pt': ('distilbert/distilbert-base-cased-distilled-squad', '564e9b5'), 'tf': ('distilbert/distilbert-base-cased-distilled-squad', '564e9b5')}}, 'type': 'text'}
    table-question-answering {'impl': <class 'transformers.pipelines.table_question_answering.TableQuestionAnsweringPipeline'>, 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForTableQuestionAnswering'>,), 'tf': (), 'default': {'model': {'pt': ('google/tapas-base-finetuned-wtq', 'e3dde19'), 'tf': ('google/tapas-base-finetuned-wtq', 'e3dde19')}}, 'type': 'text'}
    visual-question-answering {'impl': <class 'transformers.pipelines.visual_question_answering.VisualQuestionAnsweringPipeline'>, 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForVisualQuestionAnswering'>,), 'tf': (), 'default': {'model': {'pt': ('dandelin/vilt-b32-finetuned-vqa', 'd0a1f6a')}}, 'type': 'multimodal'}
    document-question-answering {'impl': <class 'transformers.pipelines.document_question_answering.DocumentQuestionAnsweringPipeline'>, 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForDocumentQuestionAnswering'>,), 'tf': (), 'default': {'model': {'pt': ('impira/layoutlm-document-qa', 'beed3c4')}}, 'type': 'multimodal'}
    fill-mask {'impl': <class 'transformers.pipelines.fill_mask.FillMaskPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForMaskedLM'>,), 'default': {'model': {'pt': ('distilbert/distilroberta-base', 'fb53ab8'), 'tf': ('distilbert/distilroberta-base', 'fb53ab8')}}, 'type': 'text'}
    summarization {'impl': <class 'transformers.pipelines.text2text_generation.SummarizationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForSeq2SeqLM'>,), 'default': {'model': {'pt': ('sshleifer/distilbart-cnn-12-6', 'a4f8f3e'), 'tf': ('google-t5/t5-small', 'df1b051')}}, 'type': 'text'}
    translation {'impl': <class 'transformers.pipelines.text2text_generation.TranslationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForSeq2SeqLM'>,), 'default': {('en', 'fr'): {'model': {'pt': ('google-t5/t5-base', 'a9723ea'), 'tf': ('google-t5/t5-base', 'a9723ea')}}, ('en', 'de'): {'model': {'pt': ('google-t5/t5-base', 'a9723ea'), 'tf': ('google-t5/t5-base', 'a9723ea')}}, ('en', 'ro'): {'model': {'pt': ('google-t5/t5-base', 'a9723ea'), 'tf': ('google-t5/t5-base', 'a9723ea')}}}, 'type': 'text'}
    text2text-generation {'impl': <class 'transformers.pipelines.text2text_generation.Text2TextGenerationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForSeq2SeqLM'>,), 'default': {'model': {'pt': ('google-t5/t5-base', 'a9723ea'), 'tf': ('google-t5/t5-base', 'a9723ea')}}, 'type': 'text'}
    text-generation {'impl': <class 'transformers.pipelines.text_generation.TextGenerationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForCausalLM'>,), 'default': {'model': {'pt': ('openai-community/gpt2', '607a30d'), 'tf': ('openai-community/gpt2', '607a30d')}}, 'type': 'text'}
    zero-shot-classification {'impl': <class 'transformers.pipelines.zero_shot_classification.ZeroShotClassificationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForSequenceClassification'>,), 'default': {'model': {'pt': ('facebook/bart-large-mnli', 'd7645e1'), 'tf': ('FacebookAI/roberta-large-mnli', '2a8f12d')}, 'config': {'pt': ('facebook/bart-large-mnli', 'd7645e1'), 'tf': ('FacebookAI/roberta-large-mnli', '2a8f12d')}}, 'type': 'text'}
    zero-shot-image-classification {'impl': <class 'transformers.pipelines.zero_shot_image_classification.ZeroShotImageClassificationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForZeroShotImageClassification'>,), 'default': {'model': {'pt': ('openai/clip-vit-base-patch32', '3d74acf'), 'tf': ('openai/clip-vit-base-patch32', '3d74acf')}}, 'type': 'multimodal'}
    zero-shot-audio-classification {'impl': <class 'transformers.pipelines.zero_shot_audio_classification.ZeroShotAudioClassificationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModel'>,), 'default': {'model': {'pt': ('laion/clap-htsat-fused', 'cca9e28')}}, 'type': 'multimodal'}
    image-classification {'impl': <class 'transformers.pipelines.image_classification.ImageClassificationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForImageClassification'>,), 'default': {'model': {'pt': ('google/vit-base-patch16-224', '3f49326'), 'tf': ('google/vit-base-patch16-224', '3f49326')}}, 'type': 'image'}
    image-feature-extraction {'impl': <class 'transformers.pipelines.image_feature_extraction.ImageFeatureExtractionPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModel'>,), 'default': {'model': {'pt': ('google/vit-base-patch16-224', '3f49326'), 'tf': ('google/vit-base-patch16-224', '3f49326')}}, 'type': 'image'}
    image-segmentation {'impl': <class 'transformers.pipelines.image_segmentation.ImageSegmentationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForImageSegmentation'>, <class 'transformers.models.auto.modeling_auto.AutoModelForSemanticSegmentation'>), 'default': {'model': {'pt': ('facebook/detr-resnet-50-panoptic', 'd53b52a')}}, 'type': 'multimodal'}
    image-to-text {'impl': <class 'transformers.pipelines.image_to_text.ImageToTextPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForVision2Seq'>,), 'default': {'model': {'pt': ('ydshieh/vit-gpt2-coco-en', '5bebf1e'), 'tf': ('ydshieh/vit-gpt2-coco-en', '5bebf1e')}}, 'type': 'multimodal'}
    object-detection {'impl': <class 'transformers.pipelines.object_detection.ObjectDetectionPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForObjectDetection'>,), 'default': {'model': {'pt': ('facebook/detr-resnet-50', '1d5f47b')}}, 'type': 'multimodal'}
    zero-shot-object-detection {'impl': <class 'transformers.pipelines.zero_shot_object_detection.ZeroShotObjectDetectionPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForZeroShotObjectDetection'>,), 'default': {'model': {'pt': ('google/owlvit-base-patch32', 'cbc355f')}}, 'type': 'multimodal'}
    depth-estimation {'impl': <class 'transformers.pipelines.depth_estimation.DepthEstimationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForDepthEstimation'>,), 'default': {'model': {'pt': ('Intel/dpt-large', 'bc15f29')}}, 'type': 'image'}
    video-classification {'impl': <class 'transformers.pipelines.video_classification.VideoClassificationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForVideoClassification'>,), 'default': {'model': {'pt': ('MCG-NJU/videomae-base-finetuned-kinetics', '488eb9a')}}, 'type': 'video'}
    mask-generation {'impl': <class 'transformers.pipelines.mask_generation.MaskGenerationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForMaskGeneration'>,), 'default': {'model': {'pt': ('facebook/sam-vit-huge', '87aecf0')}}, 'type': 'multimodal'}
    image-to-image {'impl': <class 'transformers.pipelines.image_to_image.ImageToImagePipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForImageToImage'>,), 'default': {'model': {'pt': ('caidas/swin2SR-classical-sr-x2-64', 'cee1c92')}}, 'type': 'image'}
    

 # Pipeline的創建與使用方式


```python
# %%
from transformers import pipeline, QuestionAnsweringPipeline

```

 ## 根據任務類型直接創建Pipeline, 默認都是英文的模型


```python
# %%
pipe = pipeline("text-classification")

```

    No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f (https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    Hardware accelerator e.g. GPU is available in the environment, but no `device` argument is passed to the `Pipeline` object. Model will be on CPU.
    


```python
# %%
pipe(["very good!", "vary bad!"])

```




    [{'label': 'POSITIVE', 'score': 0.9998525381088257},
     {'label': 'NEGATIVE', 'score': 0.9991207718849182}]



 ## 指定任務類型，再指定模型，創建基於指定模型的Pipeline


```python
# %%
# https://huggingface.co/models 
#https://huggingface.co/uer/roberta-base-finetuned-dianping-chinese
pipe = pipeline("text-classification", model="uer/roberta-base-finetuned-dianping-chinese")

```


```python
# %%
pipe("我覺得不太行！")

```




    [{'label': 'negative (stars 1, 2 and 3)', 'score': 0.9745733737945557}]



 ## 預先加載模型，再創建Pipeline


```python
# %%
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 這種方式，必須同時指定model和tokenizer
model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

```


```python
# %%
pipe("我覺得不太行！")

```




    [{'label': 'negative (stars 1, 2 and 3)', 'score': 0.9745732545852661}]




```python
# %%
pipe.model.device

```




    device(type='cpu')




```python
# %%
import torch
torch.cuda.is_available()

```




    True




```python
# %%
import torch
import time
#device='cpu'
times = []
for i in range(100):
    torch.cuda.synchronize() # 有安裝cuda
    
    start = time.time()
    pipe("我覺得不太行！")
    torch.cuda.synchronize() # 有安裝cuda
    end = time.time()
    times.append(end - start)
print(sum(times) / 100)

```

    0.023096537590026854
    

 ## 使用GPU進行推理


```python
# %%
pipe = pipeline("text-classification", model="uer/roberta-base-finetuned-dianping-chinese", device=0)

```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[10], line 2
          1 # %%
    ----> 2 pipe = pipeline("text-classification", model="uer/roberta-base-finetuned-dianping-chinese", device=0)
    

    File d:\pywork\ollama\transformer\prj\Lib\site-packages\transformers\pipelines\__init__.py:1164, in pipeline(task, model, config, tokenizer, feature_extractor, image_processor, processor, framework, revision, use_fast, token, device, device_map, torch_dtype, trust_remote_code, model_kwargs, pipeline_class, **kwargs)
       1161 if processor is not None:
       1162     kwargs["processor"] = processor
    -> 1164 return pipeline_class(model=model, framework=framework, task=task, **kwargs)
    

    File d:\pywork\ollama\transformer\prj\Lib\site-packages\transformers\pipelines\text_classification.py:85, in TextClassificationPipeline.__init__(self, **kwargs)
         84 def __init__(self, **kwargs):
    ---> 85     super().__init__(**kwargs)
         87     self.check_model_type(
         88         TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
         89         if self.framework == "tf"
         90         else MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
         91     )
    

    File d:\pywork\ollama\transformer\prj\Lib\site-packages\transformers\pipelines\base.py:923, in Pipeline.__init__(self, model, tokenizer, feature_extractor, image_processor, processor, modelcard, framework, task, args_parser, device, torch_dtype, binary_output, **kwargs)
        921         self.device = torch.device(f"mps:{device}")
        922     else:
    --> 923         raise ValueError(f"{device} unrecognized or not available.")
        924 else:
        925     self.device = device if device is not None else -1
    

    ValueError: 0 unrecognized or not available.



    model.safetensors:   0%|          | 0.00/409M [00:00<?, ?B/s]


    d:\pywork\ollama\transformer\prj\Lib\site-packages\huggingface_hub\file_download.py:139: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\linchao\.cache\huggingface\hub\models--uer--roberta-base-finetuned-dianping-chinese. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
    To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
      warnings.warn(message)
    


```python
# %%
pipe.model.device

```


```python
# %%
import torch
import time
times = []
for i in range(100):
    torch.cuda.synchronize()
    
    start = time.time()
    pipe("我覺得不太行！")
    torch.cuda.synchronize()
    end = time.time()
    times.append(end - start)
print(sum(times) / 100)

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[2], line 9
          6 torch.cuda.synchronize()
          8 start = time.time()
    ----> 9 pipe("我覺得不太行！")
         10 torch.cuda.synchronize()
         11 end = time.time()
    

    NameError: name 'pipe' is not defined


 ## 確定Pipeline參數


```python
# %%
qa_pipe = pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa")

```


```python
# %%
qa_pipe

```


```python
# %%
QuestionAnsweringPipeline

```


```python
# %%
qa_pipe(question="中國的首都是哪裡？", context="中國的首都是北京", max_answer_len=1)

```

 # 其他Pipeline示例


```python
# %%
from transformers import pipeline
checkpoint = "google/owlvit-base-patch32"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

```


    config.json:   0%|          | 0.00/4.42k [00:00<?, ?B/s]


    d:\pywork\ollama\transformer\prj\Lib\site-packages\huggingface_hub\file_download.py:139: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\linchao\.cache\huggingface\hub\models--google--owlvit-base-patch32. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
    To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
      warnings.warn(message)
    


    model.safetensors:   0%|          | 0.00/613M [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/775 [00:00<?, ?B/s]



    vocab.json:   0%|          | 0.00/1.06M [00:00<?, ?B/s]



    merges.txt:   0%|          | 0.00/525k [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/460 [00:00<?, ?B/s]



    preprocessor_config.json:   0%|          | 0.00/392 [00:00<?, ?B/s]



```python
from imageio import imread
from PIL import Image


image = imread('https://unsplash.com/photos/oj0zeY2Ltk4/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MTR8fHBpY25pY3xlbnwwfHx8fDE2Nzc0OTE1NDk&force=true&w=640')
im=Image.fromarray(image)
im
```

    C:\Users\linchao\AppData\Local\Temp\ipykernel_7092\1540228800.py:5: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
      image = imread('https://unsplash.com/photos/oj0zeY2Ltk4/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MTR8fHBpY25pY3xlbnwwfHx8fDE2Nzc0OTE1NDk&force=true&w=640')
    




    
![png](output_28_1.png)
    




```python
本地
import shutil
from PIL import Image
import requests

# url = "https://unsplash.com/photos/oj0zeY2Ltk4/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MTR8fHBpY25pY3xlbnwwfHx8fDE2Nzc0OTE1NDk&force=true&w=640"
# response = requests.get(url, stream=True)
# with open('img.jpg', 'wb') as out_file:
#     shutil.copyfileobj(response.raw, out_file)
# del response
im = Image.open('img2.jpg')
im
```




    
![png](output_29_0.png)
    




```python
# %%
predictions = detector(
    im,
    candidate_labels=["hat", "sunglasses", "book"],
)
predictions

```




    [{'score': 0.25893089175224304,
      'label': 'sunglasses',
      'box': {'xmin': 349, 'ymin': 228, 'xmax': 430, 'ymax': 265}},
     {'score': 0.18501578271389008,
      'label': 'book',
      'box': {'xmin': 270, 'ymin': 284, 'xmax': 502, 'ymax': 427}},
     {'score': 0.1123475655913353,
      'label': 'hat',
      'box': {'xmin': 39, 'ymin': 173, 'xmax': 260, 'ymax': 363}}]




```python
# %%
from PIL import ImageDraw

draw = ImageDraw.Draw(im)

for prediction in predictions:
    box = prediction["box"]
    label = prediction["label"]
    score = prediction["score"]
    xmin, ymin, xmax, ymax = box.values()
    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
    draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="red")

im

```




    
![png](output_31_0.png)
    



 # Pipeline背後的實現


```python
# %%
from transformers import *
import torch

```


```python
# %%
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")

```


```python
# %%
input_text = "我覺得不太行！"
inputs = tokenizer(input_text, return_tensors="pt")
inputs

```


```python
# %%
res = model(**inputs)
res

```


```python
# %%
logits = res.logits
logits = torch.softmax(logits, dim=-1)
logits

```


```python
# %%
pred = torch.argmax(logits).item()
pred

```


```python
# %%
model.config.id2label

```


```python
# %%
result = model.config.id2label.get(pred)
result

```


```python
# %%




```
