import datetime
import os

def getbakname(apath):

    modifiedTime = os.path.getmtime(apath) 
    timeStamp =  datetime.datetime.fromtimestamp(modifiedTime).strftime("%b-%d-%y-%H:%M:%S")
    #os.rename(FilePath,FilePath+"_"+timeStamp)
    return '~'+apath+'_'+timeStamp

print(getbakname('test2.py'))
