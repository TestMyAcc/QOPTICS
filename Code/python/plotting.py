# Script using matplotlib

import numpy as np
from matplotlib import pyplot as plt



def plotter(fig,X,Y,cutz, mydel2, pydel2):
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax.plot_surface(X[:,:,cutz],Y[:,:,cutz],mydel2[:,:,cutz])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title("Hand-written")
    # ax.set_zlim(np.max(mydel2[:,:,cutz])+1,np.min(mydel2[:,:,cutz])-1)
    
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.plot_surface(X[:,:,cutz],Y[:,:,cutz],pydel2[:,:,cutz])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title("Python package")
    # ax1.set_zlim(np.max(mydel2[:,:,cutz])+1,np.min(mydel2[:,:,cutz])-1)
    plt.show()
    
    return ax,ax1








# cutz = 0
# fig = plt.figure(figsize=plt.figaspect(0.5))
# # exclude edges of xy-plane
# xx = np.arange(1,Nx-1).reshape(1,Nx-2,1); 
# yy = np.arange(1,Ny-1).reshape(Ny-2,1,1);
# xx = np.arange(Nx).reshape(1,Nx,1)
# yy = np.arange(Ny).reshape(Ny,1,1)

# plotter(fig,np.squeeze(X[yy,xx,:]),np.squeeze(Y[yy,xx,:]),cutz,
#         np.squeeze(del2_F2[yy,xx,:]),np.squeeze(py_del2_F2[yy,xx,:]))






