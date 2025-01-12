---
title: vscode for hugo 
description: vscode_hugo
tags: []
categories: []
series: []
editext: md
---
目的是轉檔,rmarkdown,ipynb->markdown

## vscode 設定
<!--more-->
### vscode 總體設定
快速鍵設定的檔案位置: `C:\Users\linchao\AppData\Roaming\Code\User\keybindings.json`

內容如下:
```
[
    {
        "key": "ctrl+o ctrl+1",
        "command": "workbench.action.tasks.runTask",
        "args": "build md"
    },
    {
        "key": "ctrl+o ctrl+2",
        "command": "workbench.action.tasks.runTask",
        "args": "publish"
    },
    {
      "key": "F1",
      "command": "workbench.action.tasks.runTask",
      "args": "Build Project"
    }
]
```
### vscode 專案
vscode 專案的意思是,按滑鼠右鍵點選vscode 打開。

code 的專案目錄 裡面有一個資料夾`.vscode` ,其中一個檔案`tasks.json`
內容如下
```
{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build md",
            "type": "shell",
            //"command": "echo ${file} ${cwd} ${fileBasename} ${relativeFile}"
            "command": "rscript --encoding=UTF-8 single.r ${relativeFile}"
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
要讓上面的快速鍵作用,專案跟目錄必須有`single.r`,`convert.bat`
其中`convert.bat`的內容為:
```
@echo off
echo %1
set _filename=%~n1
set _extension=%~x1
rem echo %_filename%
rem echo %_extension%

set kind="Nomatch"
if /I %_extension%==.Rmarkdown (set kind="rmd")
if /I %_extension%==.rmd (set kind="rmd")
if /I %_extension%==.ipynb (set kind="ipynb")

if %kind%=="rmd" (rscript --encoding=UTF-8 single.r %1)
if %kind%=="ipynb" (
jupyter nbconvert %1 --to markdown --NbConvertApp.output_files_dir="%_filename%.file" 
)

if %kind%=="Nomatch" (echo unknown file type)
```
可以看到,上面的程式碼需要

- `single.r`,而single.r需要安裝`smaid`的R套件,看步驟1。
- `convert.bat` 這個則需要安裝的套件,看步驟2。jupyter,ipystata

1. Rmarkdown 轉 md
    Rmarkdown 需要 檔案 `single.r` 

    ```r
    #在R中執行
    install.packages("devtools")
    devtools::install_github("airbone4/SmdAid")
    ```
    如果需要在控制台中測試
    ```cmd
    rscript --encoding=UTF-8 single.r content\python\basic1\stata\檔案名稱.Rmarkdown"
    ```

1. ipynb轉markdown

    ```python
    pip install numpy
    pip install pandas
    pip install ipystata
    pip install jupyter
```