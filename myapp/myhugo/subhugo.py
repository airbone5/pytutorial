import os,sys,re
import shutil
import nbformat
import datetime
from nb2hugo.exporter import HugoExporter
#from nbconvert import MarkdownExporter
from nbconvert.exporters.exporter import ResourcesDict

#import json


def fixBackSlash(*astr):
 rst=[item.replace("\\","/") for item in astr]
 return rst

def isFolderHas(afolder,aname):
  alist=os.listdir(afolder)
  return aname in alist

def readText(afile):
    file = open(afile, "r",encoding='utf8')
    content=file.read().lstrip("\n")
    file.close()
    return(content)

def writeText(txt,afile):
    file = open(afile, "w",encoding='utf8')    
    file.write(txt)
    file.close()    

def realbasename(astr):
    return os.path.basename(astr).rsplit( ".", 1 )[0]

def tomd(aname,mdname):
    content=readText(aname)
    notebook=nbformat.reads(content, as_version=4)

    #notebook = notebooknode.from_dict(content)
    exporter = HugoExporter() 
    #exporter = MarkdownExporter() 
    #註解 output_files_dir, in nbconvert ,hugoexporter images_path
    #markdown, resources = exporter.from_notebook_node(notebook,resources={"images_path":os.path.basename(mdname)})
    #aResourceDict=ResourcesDict()
    #aResourceDict['images_path']="xxxx"
    #markdown, resources = exporter.from_notebook_node(notebook,aResourceDict)
    imgpath=realbasename(mdname)+'_files'
    markdown, resources = exporter.from_notebook_node(notebook,{'output_files_dir':imgpath})
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
description: auto generated by hugoconvert
weight: 300
---
"""
#config={"excludedir":["[.]git","[.]env","prj","__pycache__"]}
config={"excludedir":["prj","__pycache__","llama.cpp","checkpoints"]}

def checkAddIndex(targetfolder):
    idxcontent="""---
title: %s
description: auto
weight: 300
---
{{< local_file_list >}}
"""
    if isFolderHas(targetfolder,".skipindex"):
      return
    for folder, subfolders,filenames in os.walk(targetfolder, topdown=True):  
      tmpfolders= getExcludeDirPattern()
      tmpfolders.append('\\b.*_files\\b')
      exclude_folders='|'.join(tmpfolders)
      subfolders[:] = [d for d in subfolders if not d.startswith('.')]
      #如果目錄是被排除的目錄 ， 或者包含 .skipindex就不要加入_index.md 
      subfolders[:] = [d for d in subfolders if re.search(exclude_folders,d)==None and not isFolderHas(os.path.join(folder,d),'.skipindex')]

      if not fileIgnore(folder):
        fname=os.path.join(folder,'_index.md')
        if not os.path.exists(fname):
            atitle=os.path.basename(folder)
            atxt=idxcontent % ( atitle) 
            writeText(atxt,fname)

def fileIgnore(afile):
    rst=False
    #ignorePattern=[r'.*\\[.]git',r'.*[\\].*_files',r'.*[\\]?prj\\.*',r'.*\.mp4',r'.*[\\]?\.env',r'.*[\\]?__pycache__.*',r'.*[\\]?checkpoints\\.*',r'.*[\\]?.vscode.*']
    ignorePattern=[r'.*\.mp4']
    for p in ignorePattern:
        rst=re.fullmatch(p,afile)!=None
        if rst:
            print(f'ignore {afile}')
            break
    return(rst)



def handleres(res,targetfolder,targetname):

  for item in res["outputs"].keys():  
    
    respath=os.path.join(targetfolder,item)
    if not os.path.exists(os.path.dirname(respath)):
        os.makedirs(os.path.dirname(respath))   
    with open(respath,'wb') as f:
        f.write(res["outputs"][item])     


def getExcludeDirPattern():
    '''
    不用
    '''
    rst=[]
    for item in config["excludedir"]:
      rst.append(f"\\b{item}\\b") 
    return rst 
    

#redir=getExcludeDirPattern()

def tohugo(srcdir,dstdir):
 
    checkAddIndex(srcdir)

    #exclude_folders= '|'.join(config["excludedir"])
    exclude_folders='|'.join(getExcludeDirPattern())
    for folder, subfolders,filenames in os.walk(srcdir, topdown=True):    
      subfolders[:] = [d for d in subfolders if not d.startswith('.')]
      subfolders[:] = [d for d in subfolders if re.search(exclude_folders,d)==None ]
      # for d in subfolders:    
      #   if re.search(exclude_folders,d)!=None:
      #     print(f"忽略目錄 {os.path.join(folder,d)}")
      #     subfolders.remove(d)
      # print('提早結束')    
      # continue
      for filename in filenames:        
          srcName=os.path.join(folder,filename)
          adir=os.path.dirname(srcName)
          x=os.path.relpath(adir,srcdir) #x:原始檔案的相對目錄
          adir=os.path.join(dstdir,'content',x) #目標目錄
          dstName=os.path.join(adir,filename)        
          if fileIgnore(dstName)==False:
              if not os.path.exists(adir):
                  os.makedirs(adir)
              #print('.',end='',flush=True)
              if srcName.endswith('.ipynb'):   
                  dstName=dstName.rsplit( ".", 1 )[0]+'.md'
                  md,res=tomd(srcName,dstName) #srcname例如temp/python/basic1/tutorial.ipynb
                  #print(res)
                  handleres(res,adir,os.path.basename(srcName).rsplit('.')[0])
                  
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
