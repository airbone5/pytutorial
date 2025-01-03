"""
pyinstaller --add-data=data;. -F fixstata.py
demo:
python fixstata.py dofix -p c:/temp/fake 
 

"""
import os, sys,  zipfile
import click

 
 
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path) 

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

@click.group()
def db():
    pass

@click.command()
#@click.argument('statapath') #, prompt='new site name',  help='hugo根目錄.')
@click.option("-p","--statapath", default="",help="stata root")
def dofix(statapath):
  """
  move pysata
  """   
  if statapath=="":
      statapath="c:/program files/stata17" 
  if os.path.exists(f"{statapath}/utilities/pystata"):
      if os.path.exists(f"{statapath}/utilities/pystata.old"):
          print(f"{statapath}/utilities/pystata.old 已經存在,不做任何事結束")
          sys.exit()
      os.rename(f"{statapath}/utilities/pystata",f"{statapath}/utilities/pystata.old")
  if not os.path.exists(f"{statapath}/stata.lic"):
      print("產生lic")
      touch(f"{statapath}/stata.lic")

  with zipfile.ZipFile(resource_path('data/pystata.zip'), 'r') as z:
      for member in z.namelist():
          z.extract(member,f"{statapath}/utilities")
   
 

db.add_command(dofix)

db()