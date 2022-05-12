#%%
import os as _os
import re as _re
import h5py as _h5py

def checkdir(dirpath:str):
    dirpath = _os.path.expanduser(dirpath)
    if _os.path.isdir(dirpath) == False:
        checkdir(_os.path.dirname(dirpath))
        _os.mkdir(dirpath)
        print(f"create new dir {dirpath}\n")
        
def files(dir:str) -> list:
    dir = _os.path.expanduser(dir)
    filelist = [ f for f in _os.listdir(dir) if 
    _os.path.isfile(_os.path.join(dir, f))]
    filelist = [x.strip('.pngjpgh5') for x in filelist]
    return filelist

def ls_files(dir):
    """
    return all the filepaths in dir in a string
    """
    filenames = files(dir)
    return '\n'.join(filenames)

def ls_allfiles(dir):
    """
    return all the filepaths including 
    those in subdirs of dir in a string
    """
    filenames = ls_files(dir)
    allfilenames = [ls_files(_os.path.join(dir,f)) 
                    for f in _os.listdir(dir) 
                    if _os.path.isdir(_os.path.join(dir, f))]  
    return filenames+'\n'+'\n'.join(allfilenames)

def ls_light():
    """
    return all the names of LG files below ~/Data/
    """
    base = _os.path.expanduser("~/Data/")
    filelist = files(base)
    r = _re.compile('^LG*')
    lgfilenames =  list(filter(r.match, filelist))
    return base, '\n'.join(lgfilenames)

def ls_BEC():
    """
    return all the names of BEC files below ~/Data/
    """
    base = _os.path.expanduser("~/Data/")
    filelist = files(base)
    r = _re.compile('^BEC\S*')
    lgfilenames =  list(filter(r.match, filelist))
    return base, '\n'.join(lgfilenames)

def ls_data():
    """
    return all the names of data ~/Data/
    """
    base = _os.path.expanduser("~/Data/")
    filelist = files(base)
    r = _re.compile('^scan\S*')
    filenames =  list(filter(r.match, filelist))
    return base, '\n'.join(filenames) + \
           ls_BEC()[1] + ls_light()[1] + \
            ': (Enter to return)\n'
           


def _walk(obj, data):
    for obj_name, obj in obj.items():
        if (isinstance(obj, _h5py.Group)):        
            _walk(obj, data)
        else: data[obj_name] = obj[()]

def retrieve(datadir=r"~/Data", flag=0):
    """Calculating discrete laplacian
    Arg:
        datadir: the directory where data are stored
    specify the datadir, and choose a file to load all
        Default is "~/Data"
        
        flag: if flag is True, reading datadir directly.

    Return: dict[varnames: datasets(in hdf5)]
    """
    datadir = _os.path.expanduser(datadir)
    if flag:
        path = datadir
    else: 
        print(filenames,end='\n')
        filename = input(f"Choose a filename from below:\n{filenames}")
        filenames = ls_data()[1]
        if not (filename):
            return 
        path = _os.path.join(datadir, filename) + '.h5'
    
    

    try:
        with _h5py.File(path, "r") as f:
            datas = {}
            _walk(f, datas)
            
            if ('LGfile' in f):
                lgpath = _os.path.expanduser(f['LGfile'][()])
                if (lgpath != ''):
                    with _h5py.File(lgpath, "r") as f_lg:
                        _walk(f_lg,datas)
                        print(f"\nReading LGdata : {lgpath}\n")
            print("Retrieve data success!\n")
            
            return datas
    except FileNotFoundError as e:
        print(e)



if __name__ == "__main__":
    # checkdir(r"C:\Users\Lab\Desktop\PlayGround\dir\subdir\subsubdir")
    # print(files(r'c:\\Users\\Lab\\Data'))
    # print(ls_allfiles(r"C:\Users\Lab\Data"))
    # print(ls_light()[1])
    print(ls_data()[1])
