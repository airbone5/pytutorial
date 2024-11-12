"""
pyinstaller --add-data=data;. -F myhugo.py
demo:
python myhugo.py newsite xxx
python myhugo.py tohugo --srcdir d:/work/python --destdir xxx
"""
import os, sys,  zipfile
import click
import subhugo

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.environ.get("_MEIPASS2",os.path.abspath("."))

    return os.path.join(base_path, relative_path)
 

@click.group()
def db():
    pass

@click.command()
@click.argument('sitename') #, prompt='new site name',  help='hugo根目錄.')
def newsite(sitename):
    """
    創建hugo根目錄
    """    
    with zipfile.ZipFile(resource_path('data/base.zip'), 'r') as z:
        for member in z.namelist():
            z.extract(member,sitename)

@click.command()
@click.option("--srcdir", help="從哪裡")
@click.option("--destdir", help="到HUGO根目錄的content")
def tohugo(srcdir,destdir):
    """
    從來源(--srcdir)搬到HUGO根目錄(--destdir)中的子目錄content
    """    
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