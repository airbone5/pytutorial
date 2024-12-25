@echo off
set prj=prj
if not "%1"=="" set prj=%1
echo %prj%

if exist %prj%\Scripts\activate.bat (
  set ans=Yes 
) else (
  set ans=No
)
echo %ans% %prj% already exist
if %ans%==No (
    echo create %prj% venv
    py -m venv %prj%
)
call .\%prj%\Scripts\activate.bat
rem 已經存在不安裝
if %ans%==No (
pip install -r requirements.txt
)