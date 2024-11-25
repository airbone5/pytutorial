#import pytest
import nbformat
from nbformat import notebooknode
from nbconvert import MarkdownExporter
#from nb2hugo.exporter import HugoExporter
import json

 
def tomd(aname,mdname):
    file = open(aname, "r",encoding='utf8')
    #content=json.load(file)
    content=file.read()
    file.close()
    notebook=nbformat.reads(content, as_version=4)

    #notebook = notebooknode.from_dict(content)
    exporter = MarkdownExporter() 
    markdown, resources = exporter.from_notebook_node(notebook)
    file = open(mdname, "w",encoding='utf8')
    file.write(markdown)
    file.close()      
    return(markdown,resources)

 
    
afile=r"temp\python\basic1\tutorial.ipynb"    
md,res=tomd(afile,'xx.md')

print(res["outputs"].keys())
print(res["outputs"])

for item in res["outputs"].keys():  
  #bin=base64.urlsafe_b64decode(res["outputs"][item]) 
  #bin=base64.decodebytes (res["outputs"][item])
  with open(item,'wb') as f:
      f.write(res["outputs"][item])