#%%
# If Running in jupyter interactively would lead to suspend"
__import__("matplotlib").use("Qt5Agg")  # or use %matplotlib widget in ipython
import numpy as np
from matplotlib import pyplot as plt
import h5py
import os,dirutils
from matplotlib import cbook
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plotter1(fig,X,Y,cut, data1, data2):
    """Compare Cross-sections of two data subplots"""
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax.plot_surface(X[:,:,cut],Y[:,:,cut],data1[:,:,cut])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title("Data 1")
    # ax.set_zlim(np.max(data1[:,:,cut])+1,np.min(data1[:,:,cut])-1)
    
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.plot_surface(X[:,:,cut],Y[:,:,cut],data2[:,:,cut])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title("Data 2")
    # ax1.set_zlim(np.max(data1[:,:,cut])+1,np.min(data1[:,:,cut])-1)
    plt.show()
    
    return ax,ax1


def plotter2(fig,x, data1, data2):
    """Compare two data(array) int the same plot"""
    fig, ax = plt.subplots()
    ax.plot(x, data1, 'go', linewidth=2.0)
    ax.plot(x, data2, linewidth=2.0)
    plt.show()
    return ax



def get_demo_image():
    z = cbook.get_sample_data("axes_grid/bivariate_normal.npy", np_load=True)
    # z is a numpy array of 15x15
    return z, (-3, 4, -4, 3)


def get_test_data():
    x = np.arange(-5,6) 
    y = np.arange(-5,6) 
    xx,yy = np.meshgrid(x,y)
    zz = np.random.random(xx.shape)
    return xx,yy,zz


def add_cb(fig):
    for ax in fig.get_axes():
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        fig.add_axes(ax_cb)    
        im = ax.collections[0]
        fig.colorbar(im, cax=ax_cb)
        ax_cb.yaxis.tick_right()
        ax_cb.yaxis.set_tick_params(labelright=True)


def add_label(ax_r, label):
    axes = ax_r[0];
    axes.set_ylabel(label)


def add_slice(ax_r, coord):
    for k, ax in enumerate(ax_r):
        z = np.round(coord[k],3)
        ax.set_title(f"z={z}")

def plotting_data(ax_r, X, Y, Prop, zindice, **kwargs):
    # plotting data along each row
    for i, ax in enumerate(ax_r):
        contu = ax.pcolor(X[..., zindice[i]],Y[..., zindice[i]], Prop[...,zindice[i]], **kwargs)


def plotdata(filepath:str, nslice:int, plotwhat='intensity'):
    """Plotting the data, like comparison.m
    :filepath: path to data in h5py
    :plotwhat: 'intensity' or 'phase'
    :nslice: number of slices along z-axis
    :rtype: none
    """
    
    with h5py.File(filepath, "r") as f:
        Nx = f['Metaparameters/Nx'][()]
        Ny = f['Metaparameters/Ny'][()]
        Nz = f['Metaparameters/Nz'][()]
        Lx = f['Metaparameters/Lx'][()]
        Ly = f['Metaparameters/Ly'][()]
        Lz = f['Metaparameters/Lz'][()]
        dw = f['Metaparameters/dw'][()]
        As = f['Parameters/As'][()]
        Nbec = f['Parameters/Nbec'][()] 
        Rabi = f['Parameters/Rabi'][()]
        m = f['Parameters/m'][()]
        Wx = f['Parameters/Wx'][()]
        Wy = f['Parameters/Wy'][()]
        Wz = f['Parameters/Wz'][()]
        Ggg = f['Parameters/Ggg'][()]
        Gee = f['Parameters/Gee'][()]
        Gge = f['Parameters/Gge'][()] 
        Geg = f['Parameters/Geg'][()]
        psiG = f['psiG'][...]
        psiE = f['psiE'][...]
        lgpath = f['LGfile'][()]
        if (lgpath != ''):
            with h5py.File(lgpath, "r") as f:
                LGdata = f['LGdata'][...]
                W0 = f['Parameters/W0']
                Lambda = f['Parameters/Lambda']
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
        [X,Y,Z] = np.meshgrid(x,y,z)
        print("loading BEC succeeded!")
        
        [grid_x,grid_y,grid_z] = np.meshgrid(x,y,z)
        Epot = ( (Wx**2*grid_x**2 + Wy**2*grid_y**2 + Wz**2*grid_z**2 )
            / (2*Wz**2) )
        psiGmu = (15*Ggg / ( 16*np.pi*np.sqrt(2) )  )**(2/5)  
        psiEmu = (15*Gee / ( 16*np.pi*np.sqrt(2) )  )**(2/5) # for circular potential
        TF_amp = np.array((psiGmu-Epot)/Ggg,dtype=np.cfloat)    
        np.clip(TF_amp, 0, np.inf,out=TF_amp)
        TF_pbb = np.sqrt(TF_amp)
        total = np.sum(np.abs(TF_pbb)**2*dx*dy*dz)
        n_TF_pbb = TF_pbb/np.sqrt(total)
      
        # plotting part 
        # compute the indexes of slices along axis-z 
        middle = int(np.ceil(Nz/2))
        firstnslice = int(np.ceil(nslice/2))
        secondnslice = int(np.floor(nslice/2))
        firsthalf = np.linspace(0,middle, firstnslice, dtype=int)
        secondhalf = np.linspace(middle+np.diff(firsthalf)[-1],Nz-1, secondnslice, dtype=int)
        zindice = np.append(firsthalf, secondhalf)
    
        nrow = 3
        dims = np.zeros((nrow, nslice))
        w, h = plt.figaspect(dims)*2 
        fig, axs = plt.subplots(nrow, nslice, figsize=(w, h),tight_layout=True)
        fig.suptitle("PsiG, psiE, LG beam comparison")
        
        labels = [None]*3
        Data = np.zeros((*psiG.shape,3),dtype=np.cfloat)
        Data[...,0] = psiG
        Data[...,1] = psiE
        Data[...,2] = LG
        Profile = np.zeros_like(Data, dtype=float)
        
        if (plotwhat == 'intensity'):
            Profile[...,0:2] = np.abs(Data[...,0:2])**2*dx*dy*dz #wavefunction
            Profile[...,-1] = np.abs(Data[...,-1])**2 #Light amplitude
            labels[0:2] = ["|psiG|^2*dx*dy*dz", "|psiE|^2*dx*dy*dz"]
            labels[-1] =  "|LG|^2"

        if (plotwhat == 'phase'):
            Profile[...] = np.arctan2(np.imag(Data[...]),np.real(Data[...]))
            labels[0:2] = ["phase of psiG", "phase of psiE"]
            labels[-1] =  "phase of LG"
            

        add_slice(axs[0], z[zindice])
        for k,axs_row in enumerate(axs):
            add_label(axs_row, labels[k])
            plotting_data(axs_row, X,Y, Profile[...,k], zindice)
            
        add_cb(fig)
        plt.show()
        plt.close()



#%%
if (__name__ == '__main__'):

    base_dir = os.path.expanduser("~/Data/")
    filename = input("The filename:" + '\n' + dirutils.lsfiles(base_dir))
    path = os.path.join(base_dir,filename) + '.h5'
    
    plotdata(path,5,plotwhat='phase')
# %%
