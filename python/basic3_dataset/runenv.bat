@echo off
if exist prj\Scripts\activate.bat (
  set ans=Yes 
) else (
  set ans=No
)
echo %ans% prj already exist
if %ans%==No (
    echo create prj venv
    py -m venv prj
)
call .\prj\Scripts\activate.bat
pip install -r requirements.txt
