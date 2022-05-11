# operation realated to directory 
#%%
from fileinput import filename
import os
import re

def checkdir(dirpath:str):
    dirpath = os.path.expanduser(dirpath)
    if os.path.isdir(dirpath) == False:
        checkdir(os.path.dirname(dirpath))
        os.mkdir(dirpath)
        print(f"create new dir {dirpath}\n")
        
def files(dir:str) -> list:
    dir = os.path.expanduser(dir)
    filelist = [ f for f in os.listdir(dir) if 
    os.path.isfile(os.path.join(dir, f))]
    filelist = [x.strip('.pngjpgh5') for x in filelist]
    return filelist

def lsfiles(dir:str) -> str:
    """    return all the filepaths in dir 
    in a string"""
    filenames = files(dir)
    return '\n'.join(filenames)

def lsallfiles(dir:str) -> str:
    """    return all the filepaths including 
    those in subdirs of dir in a string"""
    filenames = lsfiles(dir)
    allfilenames = [lsfiles(f) for f in os.listdir(dir) if 
    os.path.isdir(os.path.join(dir, f))]  # reuse lsfiles
    return filenames+'\n'+'\n'.join(allfilenames)

def listLG() -> tuple[str,str]:
    """    return all the names of LG files below ~/Data/"""
    base = os.path.expanduser("~/Data/")
    filelist = files(base)
    r = re.compile('^LG*')
    lgfilenames =  list(filter(r.match, filelist))
    return base, '\n'.join(lgfilenames)

def listBEC() -> tuple[str,str]:
    base = os.path.expanduser("~/Data/")
    filelist = files(base)
    r = re.compile('^BEC\S*')
    lgfilenames =  list(filter(r.match, filelist))
    return base, '\n'.join(lgfilenames)

def listscanresults() -> tuple[str,str]:
    base = os.path.expanduser("~/Data/")
    filelist = files(base)
    r = re.compile('^scan\S*')
    lgfilenames =  list(filter(r.match, filelist))
    return base, '\n'.join(lgfilenames)

if __name__ == "__main__":
    # checkdir(r"C:\Users\Lab\Desktop\PlayGround\dir\subdir\subsubdir")
    # print(files(r'c:\\Users\\Lab\\Data'))
    # print(lsallfiles(r"C:\Users\Lab\Desktop\PlayGround"))
    # print(listLG()[1])
    print(listscanresults()[1])