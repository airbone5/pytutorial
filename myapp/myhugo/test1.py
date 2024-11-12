#import pytest
import nbformat
from nbformat import notebooknode
from nb2hugo.exporter import HugoExporter
import json

 
def tomd(aname,mdname):
    file = open(aname, "r",encoding='utf8')
    #content=json.load(file)
    content=file.read()
    print(content)
    file.close()
    notebook=nbformat.reads(content, as_version=4)

    #notebook = notebooknode.from_dict(content)
    exporter = HugoExporter() 
    markdown, resources = exporter.from_notebook_node(notebook)
    file = open(mdname, "w",encoding='utf8')
    file.write(markdown)
    file.close()      
    return(markdown,resources)

 
    
afile=r"python\basic1\tutorial.ipynb"    
md,res=tomd(afile,'xx.md')
print(md)
print(res["outputs"].keys())
