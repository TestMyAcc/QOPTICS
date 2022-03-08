# Script using matplotlib
# %%
import numpy as np
from matplotlib import pyplot as plt



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

# %%
def plotter2(fig,x, data1, data2):
    """Compare Data by plot them in two axes"""
    """Data is 1D, and in the same axes"""
    
    fig, ax = plt.subplots()
    ax.plot(x, data1, 'go', linewidth=2.0)
    ax.plot(x, data2, linewidth=2.0)
    plt.show()
    
    return ax


# %%

cut = 0
fig,ax = plt.subplots()
plotter1(fig, grid_x, grid_y, cut, TFsol, psiG)
plotter2(fig, grid_x, grid_y, TFsol, psiG)






