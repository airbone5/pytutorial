import argparse
import os,sys,re
import shutil
import nbformat
#from nbformat import notebooknode
from nb2hugo.exporter import HugoExporter
#import json


def readText(afile):
    file = open(afile, "r",encoding='utf8')
    content=file.read().lstrip("\n")
    file.close()
    return(content)

def writeText(txt,afile):
    file = open(afile, "w",encoding='utf8')    
    file.write(txt)
    file.close()    
 
def tomd(aname,mdname):
    content=readText(aname)
    notebook=nbformat.reads(content, as_version=4)

    #notebook = notebooknode.from_dict(content)
    exporter = HugoExporter() 
    markdown, resources = exporter.from_notebook_node(notebook)
    markdown=AddHugoHead(markdown,os.path.basename(aname).rsplit('.')[0])
    writeText(markdown,mdname)
    return(markdown,resources)

def hasHugoHead(afile):
    p=r"^---[\s\S]*?---"
    txt=readText(afile)
    rst= not re.match(p,txt)==None    
    return(rst)   

def AddHugoHead(atxt,atitle):
    p=r"^---[\s\S]*?---"
    rst=atxt    
    if re.match(p,atxt)==None:
        ahead=hugoHead %(atitle)
        rst=ahead+atxt    
    return(rst)

 
hugoHead="""---
title: %s
description: docker log
weight: 300
---
"""


def checkAddIndex(targetfolder):
    idxcontent="""---
title: %s
description: auto
weight: 300
---
{{< local_file_list >}}
"""
    for folder, subfolders,filenames in os.walk(targetfolder, topdown=False):   
        if not fileIgnore(folder):
            fname=os.path.join(folder,'_index.md')
            if not os.path.exists(fname):
                atitle=os.path.basename(folder)
                atxt=idxcontent % ( atitle) 
                writeText(atxt,fname)

def fileIgnore(afile):
    rst=False
    ignorePattern=[r'.*[\\]?prj\\.*',r'.*\.mp4',r'.*[\\]?\.env',r'.*[\\]?__pycache__.*',r'.*[\\]?.vscode.*']
    for p in ignorePattern:
        rst=re.fullmatch(p,afile)!=None
        if rst:
            break
    return(rst)


# main 
parser = argparse.ArgumentParser()
parser.add_argument("-s","--source",nargs=1,help ="給個python所在的子目錄") # 字串
parser.add_argument("-d","--dest", nargs=1, help=r"例如給的是base 則會寫入到base\content到哪裡")# 字串
parser.add_argument("-v", "--verbose", help="比較多的說明", action="store_true")

args = parser.parse_args()
if not len(sys.argv) > 1:
    parser.print_help()
    print(f"從 python 目錄,由-s指定,到 config.toml所在位置,例如,指定base,則檔案會被複製到base/content ")
    sys.exit()
#print(f"從 python 目錄{args.source[0]} 到 {args.dest[0]}")
srcdir=args.source[0]
dstdir=args.dest[0]

# srcdir="python"
# dstdir="base"

checkAddIndex(srcdir)

for folder, subfolders,filenames in os.walk(srcdir, topdown=False):    
    for filename in filenames:        
        srcName=os.path.join(folder,filename)
        adir=os.path.dirname(srcName)
        x=os.path.relpath(adir,srcdir)
        adir=os.path.join(dstdir,'content',x)
        dstName=os.path.join(adir,filename)        
        if fileIgnore(dstName)==False:
            if not os.path.exists(adir):
                os.makedirs(adir)
            #print('.',end='',flush=True)
            if srcName.endswith('.ipynb'):   
                dstName=dstName.rsplit( ".", 1 )[0]+'.md'
                md,res=tomd(srcName,dstName)
            else:
                if srcName.endswith('.md') or srcName.endswith('.html'):
                    if not hasHugoHead(srcName):
                        txt=readText(srcName)
                        txt=AddHugoHead(txt,os.path.basename(srcName).rsplit('.')[0])
                        print(dstName+'加入hugo表頭')
                        writeText(txt,dstName)
                    else:
                        shutil.copyfile(srcName, dstName)
                else:
                    shutil.copyfile(srcName, dstName)
