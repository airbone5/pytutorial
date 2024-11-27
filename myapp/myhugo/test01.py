import datetime
import os
def isFolderHas(afolder,aname):
  alist=os.listdir(afolder)
  return aname in alist

for folder, subfolders,filenames in os.walk("c:/pywork2/docker",topdown=True):  
  print(f"folder {folder} 次目錄 {','.join(subfolders)}")
  subfolders[:] = [d for d in subfolders if not d.startswith('.')] 
  subfolders[:]=[d for d in subfolders  if not isFolderHas(os.path.join(folder,d),'.skipindex')]
  print(f"folder {folder} has {os.listdir(folder)}")
  
  # rr=re.search(redir,folder)
  # if rr!=None:
  #   print(f"ignore {folder}")
  #   continue