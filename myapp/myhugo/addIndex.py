import os
import re



def writeText(txt,afile):
    file = open(afile, "w",encoding='utf8')    
    file.write(txt)
    file.close()    

def fileIgnore(afile):
    rst=False
    ignorePattern=[r'.*[\\]?prj\\.*',r'.*\.mp4',r'.*[\\]?\.env',r'.*[\\]?__pycache__.*',r'.*[\\]?.vscode.*']
    for p in ignorePattern:
        rst=re.fullmatch(p,afile)!=None
        if rst:
            break
    return(rst)

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


srcdir="python"
checkAddIndex(srcdir)