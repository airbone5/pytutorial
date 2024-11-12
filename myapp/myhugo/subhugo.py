import os,sys,re
import shutil
import nbformat
import datetime
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

def getbakname(apath):
    '''
    備份檔案名稱
    '''
    modifiedTime = os.path.getmtime(apath) 
    fname=os.path.basename(apath)
    
    timeStamp =  datetime.datetime.fromtimestamp(modifiedTime).strftime("%b-%d-%y-%H-%M-%S")
    fname='~'+fname+'_'+timeStamp
    fname= os.path.join( os.path.dirname(apath) ,fname)
    #os.rename(FilePath,FilePath+"_"+timeStamp)
    return fname
 
def backupfile(apath):
    aname=getbakname(apath)
    if os.path.exists(aname):
        print(aname+' replaced')
    shutil.copyfile(apath, getbakname(apath))

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

 

def tohugo(srcdir,dstdir):
    checkAddIndex(srcdir)

    for folder, subfolders,filenames in os.walk(srcdir, topdown=False):    
        for filename in filenames:        
            srcName=os.path.join(folder,filename)
            adir=os.path.dirname(srcName)
            x=os.path.relpath(adir,srcdir) #x:原始檔案的相對目錄
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


def fixcontent(srcdir):
    checkAddIndex(srcdir)

    for folder, subfolders,filenames in os.walk(srcdir, topdown=False):    
        for filename in filenames:        
            srcName=os.path.join(folder,filename)
            # adir=os.path.dirname(srcName) 
            # x=os.path.relpath(adir,srcdir) 
            # adir=os.path.join(dstdir,'content',x)
            # dstName=os.path.join(adir,filename)        
            if fileIgnore(srcName)==False:
                if srcName.endswith('.ipynb'):   
                    dstName=os.path.splitext(srcName)[0]+'.md' 
                    if os.path.exists(dstName):
                        backupfile(dstName)
                    #dstName=dstName.rsplit( ".", 1 )[0]+'.md'
                    md,res=tomd(srcName,dstName)
                else:
                    if srcName.endswith('.md') or srcName.endswith('.html'):
                        if not hasHugoHead(srcName):
                            backupfile(srcName)
                            txt=readText(srcName)
                            txt=AddHugoHead(txt,os.path.basename(srcName).rsplit('.')[0])
                            print(srcName+'加入hugo表頭')
                            writeText(txt,srcName)
