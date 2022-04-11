# operation realated to directory 
#%%
import os
# from os import listdir
# from os.path import isfile, join

def listfiles(dir:str) -> str:
    dir = os.path.expanduser(dir)
    filenames = [os.path.join(dir, f) for f in os.listdir(dir) if 
    os.path.isfile(os.path.join(dir, f))]
    return '\n'.join(filenames)
