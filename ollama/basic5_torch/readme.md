



[安裝](https://pytorch.org/)

如果有nvidia

1. 準備
  - 需要下載cuda 12.4 版
  - python 需要3.12版
1. runenv
1. pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


## requirements.txt

```
# 如果沒有CPU,下一行要comment
#--extra-index-url https://download.pytorch.org/whl/cu124
torch
torchvision
torchaudio
```



## 測試是否cuda偵測到顯示卡
```cmd
python -c "import torch; print(torch.rand(2,3).cuda())"
```
結果
```txt
✔
tensor([[0.3992, 0.0253, 0.8525],
        [0.0480, 0.4566, 0.3062]], device='cuda:0')
❌沒有正確安裝
AssertionError: Torch not compiled with CUDA enabled
```

## 查看torch 版本
```python
import torch
print(torch.__version__)
```

[PyTorch、TensorFlow和Keras，深度學習的全面比較與選擇指南](https://tw.alphacamp.co/blog/pytorch-tensorflow-keras)

暫時

[transormer and keras](https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/transformer.ipynb)