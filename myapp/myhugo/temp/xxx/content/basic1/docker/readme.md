---
title: readme
description: docker log
weight: 300
---
## 目的
這裡純粹就是一些怎樣在容器中開發的一個測試子目錄

## docker 的一些基本指令(可跳過)
1. 建立映像
```
docker build -t firstpy .
```
1. 執行
```
docker run --name first  -i -t firstpy /bin/bash
```

❓問題:
不管我們在容器裡面把print的參數字串改成甚麼 下面的指令都出現 `hello`
```
docker run -t firstpy 的執行結果都是 hello 
```
為甚麼?