---
title: 計算機
description: for phonegap
demo: html
hidden: true
---
[另一個計算機](https://medium.freecodecamp.org/how-to-build-an-html-calculator-app-from-scratch-using-javascript-4454b8714b98)


{{<  html_demo_md filePath=mycalculator.html  >}}

-:question: 如何自動設定寬高?
<!--
#body {
width: 100%;
height: 100%;
}
-->

## 第一版
{{<  html_demo_md filePath=mycalculator0.html  >}}

## 第2版
{{<  html_demo_md filePath=mycalculator1.html  >}}

## 第3版
{{<  html_demo_md filePath=mycalculator2.html  >}}

## :question:


```javascript
console.log(9*9/100)
console.log(9/100*9)
```

```
## 0.81
## 0.8099999999999999
```
正解
```
👉(9/100*9).toPrecision(15)
```
### 處理"%"

```javascript
rst= "xx".replaceAll("x","y")
console.log(rst)
```

```
## yy
```
