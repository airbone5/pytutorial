## History

### variant-learn Theme
Based on `Hugo Learn Theme`, i customized my own theme.

目前是這樣用
```
git submodule add http://u18docker:3000/linchao/myhugotheme.git themes 
```
1. 紀錄
  - add safeURL to editurl, in layout/partial/head.html
1. 刪除hug-learn.js
  hugo-learn.js裡面主要處理feather-light,這樣做的話整個site的網頁都要處理,但是有些網頁,我不要他處理。
  - 所以,我把它去掉,但是去掉的同時,裡面有個ready函數,調用了jquery.sticky()作用是保持頂板不動
  - 為了解決上面的問題,我參考[捲動](https://stackoverflow.com/questions/2731496/css-100-height-and-then-scroll-div-not-page)
    - 頂板保持不動,而其他div 可以捲動。
  - 放棄上面的作法,直接搬到`variant.js`,同時在themes的子頁面`footer.html`更改對應的hugo-learn.js為variant.js。

  - 在footer.html中註解掉modernizr.js,這個用到htmo5shiv.js 我覺得根本不用了。但是檔案都留著,

## variant
2021-12-18(23:14) :要把一個主站台加入例如rmilab.nkust.edu.tw

## 配合vscode

要配合快速鍵,則hugo 專案中的子目錄,e.g. `.vscode/tasks.json`內容如下
```
{
   // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build md",
            "type": "shell",
            //"command": "echo ${file} ${cwd} ${fileBasename} '${relativeFile}'"
            "command": "rscript --encoding=UTF-8 single.r \"${relativeFile}\""
        },
        {
            "label": "publish",
            "type": "shell",
            "command": "alarpublish",
      "dependsOrder": "sequence",
      "dependsOn": ["build md"]			
        }

    ]
}
```
hugo project 的資料夾裡面必須能夠找到single.R　，裡面會記錄infile這個變數，也就是目前要編譯的Rmarkdown。
single.R的內容很簡單,就這樣
```r
local({
  #source("themes/variant/utils/r/hugoup.r",environment(),encoding="UTF-8")
  #infile <-"content\\r\\engine\\knitr自訂輸出.Rmarkdown"
  # infile<-"content\\r\\engine\\純測試.Rmarkdown"  
  infile <- commandArgs(TRUE)
  source("themes/variant/utils/r/onepage.r",environment(),encoding="UTF-8")


},new.env())
```
single.R 呼叫　themes/variant/utils/r/onepage.r
