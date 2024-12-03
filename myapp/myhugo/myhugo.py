"""
pyinstaller --add-data=data;. -F myhugo.py
demo:
python myhugo.py newsite temp/xxx --publish_root ailearn
python myhugo.py tohugo --srcdir c:/python2 --destdir temp/xxx
python myhugo.py tohugo --srcdir c:/python2 --destdir temp/xxx --excludedir tmp 
python myhugo.py tohugo --srcdir c:/pywork2 --destdir temp/xxx -e temp -e tmp -e pretrain -e myapp


"""
import os, sys,  zipfile
import click
import subhugo

# def resource_path(relative_path):
#     """ Get absolute path to resource, works for dev and for PyInstaller """
#     try:
#         # PyInstaller creates a temp folder and stores path in _MEIPASS
#         base_path = sys._MEIPASS
#     except Exception:
#         base_path = os.environ.get("_MEIPASS2",os.path.abspath("."))

#     return os.path.join(base_path, relative_path)
 
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path) 

def collectIgnoreDir(alist):
    for item in alist:
        subhugo.config["excludedir"].append(item)

@click.group()
def db():
    pass

@click.command()
@click.argument('sitename') #, prompt='new site name',  help='hugo根目錄.')
@click.option("-p","--publish_root", default="",help="發布的根目錄")
def newsite(sitename,publish_root):
  """
  創建hugo根目錄
  """    
  #print(os.listdir(resource_path('.')))

  with zipfile.ZipFile(resource_path('data/base.zip'), 'r') as z:
      for member in z.namelist():
          z.extract(member,sitename)
  publishbat=r"""
  SET wd="%cd%"
  set outdir=_temp
  if exist %outdir% (
    rm -rf %outdir%
  ) 
  mkdir %outdir%
  hugo -s %wd% -d %outdir% -b "https://rmilab.nkust.edu.tw/public/{0}"
  xcopy /s /f /y /i "%outdir%\" "\\alar\d\rlab\www\public\{0}\"
  rm -rf %outdir%
  """
  if publish_root=="":
      publish_root=os.path.basename(sitename)
  with open(f"{sitename}/alarpublish.bat","w") as f:
      f.write(publishbat.format(publish_root))
 


@click.command()
@click.option("--srcdir", help="從哪裡")
@click.option("--destdir", help="到HUGO根目錄的content")
@click.option("-e","--excludedir",  multiple=True)
def tohugo(srcdir,destdir,excludedir):
    """    
    myhugo.py tohugo --srcdir c:/pywork2 --destdir temp/xxx  -e tmp -e pretrain -e myapp -e temp
    從來源(--srcdir)搬到HUGO根目錄(--destdir)中的子目錄content
    """       
    for item in excludedir:
      subhugo.config["excludedir"].append(item)    

    subhugo.tohugo(srcdir,destdir)

@click.command()
@click.option("--srcdir", help="目標目錄")
def fixcontent(srcdir):
    """
    補完--srcdir指令的內容(就是content section)
    """    
    subhugo.fixcontent(srcdir)


db.add_command(newsite)
db.add_command(tohugo)
db.add_command(fixcontent)
db()