
#%%
# __import__("matplotlib").use("Qt5Agg")  # or use %matplotlib in ipython
import numpy as np
from matplotlib import pyplot as plt
import h5py
import os


def plotter1(fig,X,Y,cut, data1, data2):
    """Compare Data by plot them in two subplots"""
    """Data is 3D, need cut"""
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
    """Compare Data by plot them in two axes"""
    """Data is 1D, and in the same axes"""
    
    fig, ax = plt.subplots()
    ax.plot(x, data1, 'go', linewidth=2.0)
    ax.plot(x, data2, linewidth=2.0)
    plt.show()
        
    return ax

def plotting_data(ax_r, X, Y, Data, zindice, **kwargs):
    for i, ax in enumerate(ax_r):
        contour = ax.pcolor(X[..., zindice[i]],Y[..., zindice[i]], Data[...,zindice[i]], **kwargs)
        return contour
    

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
        Wx = f['Parameters/Wx'][()]
        Wy = f['Parameters/Wy'][()]
        Wz = f['Parameters/Wz'][()]
        Ggg = f['Parameters/Ggg'][()]
        Gee = f['Parameters/Gee'][()]
        Gge = f['Parameters/Gge'][()] 
        Geg = f['Parameters/Geg'][()]
        psiG = f['psiG'][...]
        psiE = f['psiE'][...]
        # lgpath = f['LGfile'][()]
        lgpath=''
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
        
        Data = np.zeros((*psiG.shape,3),dtype=np.cfloat)
        Data[...,0] = psiG
        Data[...,1] = psiE
        Data[...,2] = LG
        Profile = np.zeros_like(Data, dtype=float)
        
        if (plotwhat == 'intensity'):
            Profile[...,0:2] = np.abs(Data[...,0:2])**2*dx*dy*dz #wavefunction
            Profile[...,-1] = np.abs(Data[...,-1])**2 #Light amplitude
        if (plotwhat == 'phase'):
            Profile[...] = np.arctan2(np.imag(Data[...]),np.real(Data[...]))
            
        for k,axs_row in enumerate(axs):
            img = plotting_data(axs_row, X,Y, Profile[...,k], zindice)
            
            # fig.subplots_adjust(right=0.1)
            # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            # fig.colorbar(img, cax=cbar_ax)
            
        # for col,zidx in enumerate(zindice):
        #     axs[0, col].pcolor(
        #         X[..., z[zidx]],Y[..., z[zidx]], np.abs(n_TF_pbb[..., z[zidx]])**2*dx*dy*dz)
        # for col,zidx in enumerate(zindice):
        #     axs[1, col].pcolor(
        #         X[..., z[zidx]],Y[..., z[zidx]], np.abs(psiG[..., z[zidx]])**2*dx*dy*dz)
        # for col,zidx in enumerate(zindice):
        #     axs[2, col].pcolor(
        #         X[..., z[zidx]],Y[..., z[zidx]], np.abs(psiE[..., z[zidx]])**2*dx*dy*dz)
        # elif (plotwhat == 'phase'):
        #     for col,zidx in enumerate(zindice):
        #         phase = np.arctan2(np.imag(psiG[:,:,cut])
        #                             ,np.real(psiG[:,:,cut]))
        #         axs[0, col].pcolor(
        #             X[..., z[zidx]],Y[..., z[zidx]], np.abs(n_TF_pbb[..., z[zidx]])**2*dx*dy*dz)
        #     for col,zidx in enumerate(zindice):
        #         axs[1, col].pcolor(
        #             X[..., z[zidx]],Y[..., z[zidx]], np.abs(psiG[..., z[zidx]])**2*dx*dy*dz)
        #     for col,zidx in enumerate(zindice):
        #         axs[2, col].pcolor(
        #             X[..., z[zidx]],Y[..., z[zidx]], np.abs(psiE[..., z[zidx]])**2*dx*dy*dz)
            
        plt.show()
        # plt.close()

def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

def annotate_axes(ax, text, fontsize=18):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="darkgrey")

#%%
if (__name__ == '__main__'):
    import numpy as np
    from matplotlib import pyplot as plt
    import h5py
    # plt.switch_backend('QtAgg4')

    path = 'c:\\Users\\Lab\\Desktop\\Data\\local\\0331\\BEC_LP10_121-121-121.h5'
    
    plotdata(path,5,plotwhat='intensity') 
    
    # plt.close('all')
    # fig, ax = plt.subplots()
    # example_plot(ax, fontsize=30)
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, tight_layout=False)
    # example_plot(ax1)
    # example_plot(ax2)
    # example_plot(ax3)
    # example_plot(ax4)
    
    
    # fig = plt.figure()
    # fig.add_subplot(231)
    # ax1 = fig.add_subplot(2, 3, 1)  # equivalent but more general

    # fig.add_subplot(232, frameon=False)  # subplot with no frame
    # fig.add_subplot(233, projection='polar')  # polar subplot
    # fig.add_subplot(234, sharex=ax1)  # subplot sharing x-axis with ax1
    # fig.add_subplot(235, facecolor="red")  # red subplot

    # ax1.remove()  # delete ax1 from the figure
    # fig.add_subplot(ax1)  # add ax1 back to the figure
                
    
    # fig, axd = plt.subplot_mosaic([['upper left', 'upper right'],
    #                            ['lower left', 'lower right']],
    #                           figsize=(5.5, 3.5), constrained_layout=True)
    # for k in axd:
    #     annotate_axes(axd[k], f'axd["{k}"]', fontsize=14)
    # fig.suptitle('plt.subplot_mosaic()')
    # cut = 0
    # fig,ax = plt.subplots()
    # plotter1(fig, grid_x, grid_y, cut, TFsol, psiG)
    # plotter2(fig, grid_x, grid_y, TFsol, psiG)




# %%

# %%
