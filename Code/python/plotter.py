#%%
# If Running in jupyter interactively would lead to suspend"
# __import__("matplotlib").use("TkAgg")  # or use %matplotlib widget in ipython
import numpy as np
from matplotlib import pyplot as plt
import h5py
import os,dirutils
from matplotlib import cbook
from mpl_toolkits.axes_grid1 import make_axes_locatable
import current as cur
import oam
import scipy.io


def add_cb(fig):
    for ax in fig.get_axes():
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        fig.add_axes(ax_cb)    
        im = ax.collections[0]
        fig.colorbar(im, cax=ax_cb)
        ax_cb.yaxis.tick_right()
        ax_cb.yaxis.set_tick_params(labelright=True)


def add_label(axs, coord, labels):
    for k, ax_r in enumerate(axs):
        ax_r[0].set_ylabel(labels[k])
        if not (hasattr(ax_r, '__iter__')):
            z = np.round(coord,3)
            ax_r.set_title(f"z={z}")
        else:
            for m, ax in enumerate(ax_r):
                z = np.round(coord[m],3)
                ax.set_title(f"z={z}")
            
def add_contour(axs, profile, X, Y, zindice):
    for k,axs_row in enumerate(axs):
        plotted = profile[...,k]
        for i, ax in enumerate(axs_row):
            ax.pcolor(X[..., zindice[i]],Y[..., zindice[i]], plotted[...,zindice[i]])
    add_cb(axs[0,0].get_figure())


def plotting(axs,z, zindice, X, Y, Data, dx,dy,dz, plotwhat):
    labels = [None]*3
    profile = np.zeros_like(Data, dtype=float)
    if (plotwhat == 'intensity'):
        profile[...,1:] = np.abs(Data[...,1:])**2*dx*dy*dz #wavefunction
        profile[...,0] = np.abs(Data[...,0])**2 #Light amplitude
        labels[0] =  "|LG|^2"
        labels[1:] = ["|psiE|^2*dx*dy*dz", "|psiG|^2*dx*dy*dz"]

    if (plotwhat == 'phase'):
        profile[...] = np.arctan2(np.imag(Data[...]),np.real(Data[...]))
        labels[0] =  "phase of LG"
        labels[1:] = ["phase of psiE", "phase of psiG"]
    add_contour(axs, profile, X, Y, zindice)
    return labels


def plotarrow(axs,zindice,X,Y,dx,dy,dz,m,length,data,**quiverargs):
    # probability density current and linear momentum of light
    Jx = Jy = Jz = np.zeros_like(data[...,1],dtype=np.complex128)
    s = quiverargs.pop('step',1) #default to 1, no lost arrow

    idx = np.arange(0,X.shape[1],s)
    idy = np.arange(0,Y.shape[0],s)
    kernel = np.ix_(idx,idy)
    X = X[kernel]
    Y = Y[kernel]

    for k,axs_row in enumerate(axs):
        if (k==1 or k==2):
            Jx,Jy,Jz = cur.current(data[...,k],dx,dy,dz,m)
        else:
            Jx,Jy,Jz = oam.oam(data[...,k],dx,dy,dz,length)        
       
        # reduce the density of arrow on axes
        Jx = Jx[kernel]
        Jy = Jy[kernel]

        for i, ax in enumerate(axs_row):
            ax.quiver(X[..., zindice[i]], Y[..., zindice[i]], 
                    np.real(Jx[..., zindice[i]]), np.real(Jy[..., zindice[i]]),  
                    **quiverargs)
    labels = [None]*3
    labels[0] =  "O.A.M of L.G"
    labels[1:] = ["P.D.C of psiE", "P.D.C of psiG"]
    return labels

def plotdata(filepath:str, nslice:int, plotwhat='phase', current=False,**quiverargs):
    """Plotting the data, like comparison.m
    :filepath: path to data in h5py
    :plotwhat: 'intensity' or 'phase'
    :nslice: number of slices along z-axis
    :rtype: none
    """
    
    # just retrieve data and do some calculation
    with h5py.File(filepath, "r") as f:
        Nx = f['Metaparameters/Nx'][()]
        Ny = f['Metaparameters/Ny'][()]
        Nz = f['Metaparameters/Nz'][()]
        Lx = f['Metaparameters/Lx'][()]
        Ly = f['Metaparameters/Ly'][()]
        Lz = f['Metaparameters/Lz'][()]
        # dw = f['Metaparameters/dw'][()]
        # As = f['Parameters/As'][()]
        # Nbec = f['Parameters/Nbec'][()] 
        Rabi = f['Parameters/Rabi'][()]
        m = f['Parameters/m'][()]
        Wx = f['Parameters/Wx'][()]
        Wy = f['Parameters/Wy'][()]
        Wz = f['Parameters/Wz'][()]
        Ggg = f['Parameters/Ggg'][()]
        # Gee = f['Parameters/Gee'][()]
        # Gge = f['Parameters/Gge'][()] 
        # Geg = f['Parameters/Geg'][()]
        psiG = f['psiG'][...]
        psiE = f['psiE'][...]
        lgpath = f['LGfile'][()]
        
        lgpath = os.path.expanduser(lgpath)
        if (lgpath != ''):
            with h5py.File(lgpath, "r") as f:
                LGdata = f['LGdata'][...]
                # W0 = f['Parameters/W0']
                Lambda = f['Parameters/Lambda'][()]
                Rabi = Rabi/Wz                                                                               
                LG = 0.5*Rabi*LGdata
                print(f"\nReading LGdata : {lgpath}\n")
        else:
            print(f"\n{lgpath} doesn't exits!\nset LG = 0, lgpath to ''\n")
            LG = 0
            lgpath=''   
    
        
        x = np.linspace(-Lx,Lx,Nx)
        y = np.linspace(-Ly,Ly,Ny)
        z = np.linspace(-Lz,Lz,Nz)
        dx = np.diff(x)[0]
        dy = np.diff(y)[0]
        dz = np.diff(z)[0]
        [X,Y,_] = np.meshgrid(x,y,z)
        print("loading BEC succeeded!")
        
        [grid_x,grid_y,grid_z] = np.meshgrid(x,y,z)
        Epot = ( (Wx**2*grid_x**2 + Wy**2*grid_y**2 + Wz**2*grid_z**2 )
            / (2*Wz**2) )
        psiGmu = (15*Ggg / ( 16*np.pi*np.sqrt(2) )  )**(2/5)  
        # psiEmu = (15*Gee / ( 16*np.pi*np.sqrt(2) )  )**(2/5)
        TF_amp = np.array((psiGmu-Epot)/Ggg,dtype=np.cfloat)    
        np.clip(TF_amp, 0, np.inf,out=TF_amp)
        TF_pbb = np.sqrt(TF_amp)
        # total = np.sum(np.abs(TF_pbb)**2*dx*dy*dz)
        # n_TF_pbb = TF_pbb/np.sqrt(total)
      
      
        # compute the indexes of z-slices, must include center
        middle = int(np.floor(Nz/2))
        firstnslice = int(np.ceil(nslice/2))
        secondnslice = int(np.floor(nslice/2))
        firsthalf = np.linspace(0,middle, firstnslice, dtype=int)
        secondhalf = np.linspace(middle+1,Nz-1, secondnslice, dtype=int)
        zindice = np.append(firsthalf, secondhalf)
        nrow = 3
        dims = np.zeros((nrow, nslice))
        w, h = plt.figaspect(dims)*2 
        fig, axs = plt.subplots(nrow, nslice, figsize=(w, h),tight_layout=True)
        fig.suptitle("PsiG, psiE, LG beam comparison")

        Data = np.zeros((*psiG.shape,3),dtype=np.cfloat)
        # mat = scipy.io.loadmat(r'C:\Users\Lab\Documents\121_121_121W01_Lambda1_L&P10.mat')
        # Data[...,0] = mat['LG']
        # print(r"reading C:\Users\Lab\Documents\121_121_121W01_Lambda1_L&P10.mat")
        Data[...,0] = LG 
        Data[...,1] = psiE
        Data[...,2] = psiG 

        if not (plotwhat == 'none'):
            labels = plotting(axs,z,zindice, X,Y,Data,dx,dy,dz,plotwhat)
            add_label(axs, z[zindice], labels)
            
        if (current):
            labels = plotarrow(axs,zindice,X,Y, dx,dy,dz,m,Lambda, Data,**quiverargs)
            if (plotwhat == 'none'):
                add_label(axs, z[zindice], labels)
            
        
        plt.show()          # interactive backend
        # plt.close()



#%%
if (__name__ == '__main__'):

    base_dir, text  = dirutils.listBEC()
    print(text)
    filename = input("The filename:" + '\n' + text)
    path = os.path.join(base_dir,filename) + '.h5'
    plotdata(path,4, plotwhat='none',current=True,
            angles='xy',
            units='x',
            step = 5,
            scale_units='x',
            # scale=0.001,
            # headwidth=2,
            # headlength=6,
            # headaxislength=3,
            # width=0.1
            )



# %%
