import datetime
import os

for folder, subfolders,filenames in os.walk("c:/pywork2",topdown=True):  
  subfolders[:] = [d for d in subfolders if not d.startswith('.')]
  print(folder,filenames)
  
  # rr=re.search(redir,folder)
  # if rr!=None:
  #   print(f"ignore {folder}")
  #   continue