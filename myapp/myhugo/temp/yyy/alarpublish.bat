
SET wd="%cd%"
set outdir=d:\temp\fakeout
if exist %outdir% (
  rm -rf %outdir%
) 
mkdir %outdir%
hugo -s %wd% -d %outdir% -b "https://rmilab.nkust.edu.tw/public/python/"
xcopy /s /f /y "%outdir%\" "\\alar\d\rlab\www\public\python\"
rem ssh -t alar sftp -r linchao@tali:/d:/work/fakeout/* ~/local/docker/rlab/caddy/www

