#%%
# If Running in jupyter interactively would lead to suspend"
# __import__("matplotlib").use("TkAgg")  # or use %matplotlib widget in ipython
import sys
import numpy as _np
from matplotlib import pyplot as _plt
import dirutils as _dirutils
from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_locatable
import current as _cur
import oam as _oam
from dummyconst import *  # dirty step to initalize vars. no encapsulation.

def __add_cb(fig):
    for ax in fig.get_axes():
        divider = _make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        fig.add_axes(ax_cb)    
        im = ax.collections[0]
        fig.colorbar(im, cax=ax_cb)
        ax_cb.yaxis.tick_right()
        ax_cb.yaxis.set_tick_params(labelright=True)


def __add_label(axs, coord, labels):
    for k, ax_r in enumerate(axs):
        ax_r[0].set_ylabel(labels[k])
        if not (hasattr(ax_r, '__iter__')):
            z = _np.round(coord,3)
            ax_r.set_title(f"z={z}")
        else:
            for m, ax in enumerate(ax_r):
                z = _np.round(coord[m],3)
                ax.set_title(f"z={z}")


def __add_textbox(axs, param_str):
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    pos = axs[0,0].get_position().get_points()
    left = pos[0,0]
    axs[0,0].text(left, 0.95, param_str, transform=axs[0,0].figure.transFigure, fontsize=14,
        verticalalignment='top', bbox=props)
    # axs[0,0].text(0, 0.95, param_str, transform=axs[0,0].transAxes, fontsize=14,
    # verticalalignment='top', bbox=props)

            
def __add_contour(axs, profile, X, Y, zindice):
    for k,axs_row in enumerate(axs):
        plotted = profile[...,k]
        for i, ax in enumerate(axs_row):
            ax.pcolor(X[..., zindice[i]],Y[..., zindice[i]], plotted[...,zindice[i]])
    __add_cb(axs[0,0].get_figure())


def __plotting(axs,z, zindice, X, Y, Data, dx,dy,dz, plotwhat):
    labels = [None]*3
    profile = _np.zeros_like(Data, dtype=float)
    if (plotwhat == 'intensity'):
        profile[...,1:] = _np.abs(Data[...,1:])**2*dx*dy*dz #wavefunction
        profile[...,0] = _np.abs(Data[...,0])**2 #Light amplitude
        labels[0] =  "|LG|^2"
        labels[1:] = ["|psiE|^2*dx*dy*dz", "|psiG|^2*dx*dy*dz"]

    if (plotwhat == 'phase'):
        profile[...] = _np.arctan2(_np.imag(Data[...]),_np.real(Data[...]))
        labels[0] =  "phase of LG"
        labels[1:] = ["phase of psiE", "phase of psiG"]
    __add_contour(axs, profile, X, Y, zindice)
    return labels


def __plotarrow(axs,zindice,X,Y,dx,dy,dz,m,length,data,**quiverargs):
    # probability density current and linear momentum of light
    Jx = Jy = Jz = _np.zeros_like(data[...,1],dtype=_np.complex128)
    s = quiverargs.pop('step',1) #default to 1, no lost arrow

    idx = _np.arange(0,X.shape[1],s)
    idy = _np.arange(0,Y.shape[0],s)
    kernel = _np.ix_(idx,idy)
    X = X[kernel]
    Y = Y[kernel]

    for k,axs_row in enumerate(axs):
        if (k==1 or k==2):
            Jx,Jy,Jz = _cur.current(data[...,k],dx,dy,dz,m)
        else:
            Jx,Jy,Jz = _oam.oam(data[...,k],dx,dy,dz,length)        
       
        # reduce the density of arrow on axes
        Jx = Jx[kernel]
        Jy = Jy[kernel]

        for i, ax in enumerate(axs_row):
            ax.quiver(X[..., zindice[i]], Y[..., zindice[i]], 
                    _np.real(Jx[..., zindice[i]]), _np.real(Jy[..., zindice[i]]),  
                    **quiverargs)
    labels = [None]*3
    labels[0] =  "O.A.M of L.G"
    labels[1:] = ["P.D.C of psiE", "P.D.C of psiG"]
    return labels

def plotdata(nslice:int, plotwhat='phase', current=False,**quiverargs):
    """plotting the data, like comparison.m
        
       plotdata(filepath:str, nslice:int, plotwhat='phase', current=False,**quiverargs)
    
    :filepath: path to data in h5py
    :nslice: number of slices along z-axis
    :plotwhat: 'intensity' or 'phase'
    :rtype: none
    """
   
    
    data = _dirutils.retrieve()
    if not (data):
        return
    module = sys.modules[__name__]
    for name, value in data.items():
        setattr(module, name, value)
    if 'LGdata' in globals(): # The light is stored as variable named 'LGdata' in previous time.
        setattr(module, 'LG', LGdata)
    
    x = _np.linspace(-Lx,Lx,Nx)
    y = _np.linspace(-Ly,Ly,Ny)
    z = _np.linspace(-Lz,Lz,Nz)
    dx = _np.diff(x)[0]
    dy = _np.diff(y)[0]
    dz = _np.diff(z)[0]
        
    [X,Y,_] = _np.meshgrid(x,y,z)
    
    [grid_x,grid_y,grid_z] = _np.meshgrid(x,y,z)
    Epot = ( (Wx**2*grid_x**2 + Wy**2*grid_y**2 + Wz**2*grid_z**2 )
        / (2*Wz**2) )
    psiGmu = (15*Ggg / ( 16*_np.pi*_np.sqrt(2) )  )**(2/5)  
    psiEmu = (15*Gee / ( 16*_np.pi*_np.sqrt(2) )  )**(2/5)
    TF_amp = _np.array((psiGmu-Epot)/Ggg,dtype=_np.cfloat)    
    _np.clip(TF_amp, 0, _np.inf,out=TF_amp)
    TF_pbb = _np.sqrt(TF_amp)
    total = _np.sum(_np.abs(TF_pbb)**2*dx*dy*dz)
    n_TF_pbb = TF_pbb/_np.sqrt(total)
    
    
    # compute the indexes of z-slices, must include center
    middle = int(_np.floor(Nz/2))
    firstnslice = int(_np.ceil(nslice/2))
    secondnslice = int(_np.floor(nslice/2))
    firsthalf = _np.linspace(1,middle, firstnslice, dtype=int) #exclude boundary
    secondhalf = _np.linspace(middle+1,Nz-2, secondnslice, dtype=int) # exclude boundary
    zindice = _np.append(firsthalf, secondhalf)
    nrow = 3
    dims = _np.zeros((nrow, nslice))
    w, h = _plt.figaspect(dims)*2 
    fig, axs = _plt.subplots(nrow, nslice, figsize=(w, h),tight_layout=True)
    fig.suptitle("PsiG, psiE, LG beam comparison")


    Data = _np.zeros((*psiG.shape,3),dtype=_np.cfloat)


    Data[...,0] = LG 
    Data[...,1] = psiE
    Data[...,2] = psiG 

    if not (plotwhat == 'none'):
        labels = __plotting(axs,z,zindice, X,Y,Data,dx,dy,dz,plotwhat)
        __add_label(axs, z[zindice], labels)
        
    if (current):
        labels = __plotarrow(axs,zindice,X,Y, dx,dy,dz,m,Lambda, Data,**quiverargs)
        if (plotwhat == 'none'):
            __add_label(axs, z[zindice], labels)
    

    textstr = ', '.join((
        r'$W_{x}=%.2f$' % (Wx, ),
        r'$W_{y}=%.2f$' % (Wy, ),
        r'$W_{z}=%.2f$' % (Wz, ),
        r'$L=%d$' % (1, ),
        r'$N=%d$' % (1, ),
        # r'$\mathrm{median}=%.2f$' % (median, ),
        # r'$\sigma=%.2f$' % (sigma, )))
    ))
    __add_textbox(axs, textstr)
    
    _plt.show()          # interactive backend
    # _plt.close()



#%%
if (__name__ == '__main__'):

    
    plotdata(8, plotwhat='intensity',current=False,
            # angles='xy',
            # units='x',
            step = 6,
            # scale_units='x',
            # scale=0.001,
            headwidth=2,
            headlength=6,
            headaxislength=3,
            # width=0.1
            )

# %%
#%%
# If Running in jupyter interactively would lead to suspend"
# __import__("matplotlib").use("TkAgg")  # or use %matplotlib widget in ipython
import sys
import numpy as _np
from matplotlib import pyplot as _plt
import dirutils as _dirutils
from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_locatable
import current as _cur
import oam as _oam
from dummyconst import *  # dirty step to initalize vars. no encapsulation.

def __add_cb(fig):
    for ax in fig.get_axes():
        divider = _make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        fig.add_axes(ax_cb)    
        im = ax.collections[0]
        fig.colorbar(im, cax=ax_cb)
        ax_cb.yaxis.tick_right()
        ax_cb.yaxis.set_tick_params(labelright=True)


def __add_label(axs, coord, labels):
    for k, ax_r in enumerate(axs):
        ax_r[0].set_ylabel(labels[k])
        if not (hasattr(ax_r, '__iter__')):
            z = _np.round(coord,3)
            ax_r.set_title(f"z={z}")
        else:
            for m, ax in enumerate(ax_r):
                z = _np.round(coord[m],3)
                ax.set_title(f"z={z}")


def __add_textbox(axs, param_str):
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    pos = axs[0,0].get_position().get_points()
    left = pos[0,0]
    axs[0,0].text(left, 0.95, param_str, transform=axs[0,0].figure.transFigure, fontsize=14,
        verticalalignment='top', bbox=props)
    # axs[0,0].text(0, 0.95, param_str, transform=axs[0,0].transAxes, fontsize=14,
    # verticalalignment='top', bbox=props)

            
def __add_contour(axs, profile, X, Y, zindice):
    for k,axs_row in enumerate(axs):
        plotted = profile[...,k]
        for i, ax in enumerate(axs_row):
            ax.pcolor(X[..., zindice[i]],Y[..., zindice[i]], plotted[...,zindice[i]])
    __add_cb(axs[0,0].get_figure())


def __plotting(axs,z, zindice, X, Y, Data, dx,dy,dz, plotwhat):
    labels = [None]*3
    profile = _np.zeros_like(Data, dtype=float)
    if (plotwhat == 'intensity'):
        profile[...,1:] = _np.abs(Data[...,1:])**2*dx*dy*dz #wavefunction
        profile[...,0] = _np.abs(Data[...,0])**2 #Light amplitude
        labels[0] =  "|LG|^2"
        labels[1:] = ["|psiE|^2*dx*dy*dz", "|psiG|^2*dx*dy*dz"]

    if (plotwhat == 'phase'):
        profile[...] = _np.arctan2(_np.imag(Data[...]),_np.real(Data[...]))
        labels[0] =  "phase of LG"
        labels[1:] = ["phase of psiE", "phase of psiG"]
    __add_contour(axs, profile, X, Y, zindice)
    return labels


def __plotarrow(axs,zindice,X,Y,dx,dy,dz,m,length,data,**quiverargs):
    # probability density current and linear momentum of light
    Jx = Jy = Jz = _np.zeros_like(data[...,1],dtype=_np.complex128)
    s = quiverargs.pop('step',1) #default to 1, no lost arrow

    idx = _np.arange(0,X.shape[1],s)
    idy = _np.arange(0,Y.shape[0],s)
    kernel = _np.ix_(idx,idy)
    X = X[kernel]
    Y = Y[kernel]

    for k,axs_row in enumerate(axs):
        if (k==1 or k==2):
            Jx,Jy,Jz = _cur.current(data[...,k],dx,dy,dz,m)
        else:
            Jx,Jy,Jz = _oam.oam(data[...,k],dx,dy,dz,length)        
       
        # reduce the density of arrow on axes
        Jx = Jx[kernel]
        Jy = Jy[kernel]

        for i, ax in enumerate(axs_row):
            ax.quiver(X[..., zindice[i]], Y[..., zindice[i]], 
                    _np.real(Jx[..., zindice[i]]), _np.real(Jy[..., zindice[i]]),  
                    **quiverargs)
    labels = [None]*3
    labels[0] =  "O.A.M of L.G"
    labels[1:] = ["P.D.C of psiE", "P.D.C of psiG"]
    return labels

def plotdata(nslice:int, plotwhat='phase', current=False,**quiverargs):
    """plotting the data, like comparison.m
        
       plotdata(filepath:str, nslice:int, plotwhat='phase', current=False,**quiverargs)
    
    :filepath: path to data in h5py
    :nslice: number of slices along z-axis
    :plotwhat: 'intensity' or 'phase'
    :rtype: none
    """
   
    
    data = _dirutils.retrieve()
    if not (data):
        return
    module = sys.modules[__name__]
    for name, value in data.items():
        setattr(module, name, value)
    if 'LGdata' in globals(): # The light is stored as variable named 'LGdata' in previous time.
        setattr(module, 'LG', LGdata)
    
    x = _np.linspace(-Lx,Lx,Nx)
    y = _np.linspace(-Ly,Ly,Ny)
    z = _np.linspace(-Lz,Lz,Nz)
    dx = _np.diff(x)[0]
    dy = _np.diff(y)[0]
    dz = _np.diff(z)[0]
        
    [X,Y,_] = _np.meshgrid(x,y,z)
    
    [grid_x,grid_y,grid_z] = _np.meshgrid(x,y,z)
    Epot = ( (Wx**2*grid_x**2 + Wy**2*grid_y**2 + Wz**2*grid_z**2 )
        / (2*Wz**2) )
    psiGmu = (15*Ggg / ( 16*_np.pi*_np.sqrt(2) )  )**(2/5)  
    psiEmu = (15*Gee / ( 16*_np.pi*_np.sqrt(2) )  )**(2/5)
    TF_amp = _np.array((psiGmu-Epot)/Ggg,dtype=_np.cfloat)    
    _np.clip(TF_amp, 0, _np.inf,out=TF_amp)
    TF_pbb = _np.sqrt(TF_amp)
    total = _np.sum(_np.abs(TF_pbb)**2*dx*dy*dz)
    n_TF_pbb = TF_pbb/_np.sqrt(total)
    
    
    # compute the indexes of z-slices, must include center
    middle = int(_np.floor(Nz/2))
    firstnslice = int(_np.ceil(nslice/2))
    secondnslice = int(_np.floor(nslice/2))
    firsthalf = _np.linspace(1,middle, firstnslice, dtype=int) #exclude boundary
    secondhalf = _np.linspace(middle+1,Nz-2, secondnslice, dtype=int) # exclude boundary
    zindice = _np.append(firsthalf, secondhalf)
    nrow = 3
    dims = _np.zeros((nrow, nslice))
    w, h = _plt.figaspect(dims)*2 
    fig, axs = _plt.subplots(nrow, nslice, figsize=(w, h),tight_layout=True)
    fig.suptitle("PsiG, psiE, LG beam comparison")


    Data = _np.zeros((*psiG.shape,3),dtype=_np.cfloat)


    Data[...,0] = LG 
    Data[...,1] = psiE
    Data[...,2] = psiG 

    if not (plotwhat == 'none'):
        labels = __plotting(axs,z,zindice, X,Y,Data,dx,dy,dz,plotwhat)
        __add_label(axs, z[zindice], labels)
        
    if (current):
        labels = __plotarrow(axs,zindice,X,Y, dx,dy,dz,m,Lambda, Data,**quiverargs)
        if (plotwhat == 'none'):
            __add_label(axs, z[zindice], labels)
    

    textstr = ', '.join((
        r'$W_{x}=%.2f$' % (Wx, ),
        r'$W_{y}=%.2f$' % (Wy, ),
        r'$W_{z}=%.2f$' % (Wz, ),
        r'$L=%d$' % (1, ),
        r'$N=%d$' % (1, ),
        # r'$\mathrm{median}=%.2f$' % (median, ),
        # r'$\sigma=%.2f$' % (sigma, )))
    ))
    __add_textbox(axs, textstr)
    
    _plt.show()          # interactive backend
    # _plt.close()



#%%
if (__name__ == '__main__'):

    
    plotdata(8, plotwhat='intensity',current=False,
            # angles='xy',
            # units='x',
            step = 6,
            # scale_units='x',
            # scale=0.001,
            headwidth=2,
            headlength=6,
            headaxislength=3,
            # width=0.1
            )

# %%
