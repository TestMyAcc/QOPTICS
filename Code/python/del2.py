# 3D numpy array example:
# memory representation:
#     np.array([[[ dim 3 ], ... dim 2 ... ]], ... dim 1 ... , [ ... ]]])
# In the memory order, index of last dimension change first.

from scipy.ndimage import laplace
import numpy as np
import time

def timeit(fun,dx,dy,dz):
    """Compare execution time of different implementation"""
    start_time1 = time.time()
    vectorized(fun,dx,dy,dz)
    print("\n---hand-written in %s seconds ---\n" % (time.time() - start_time1))



def vectorized(F, dx, dy, dz):
    """Calculating discrete laplacian
    
    # TODO: order of accuracy.
    
    Arg:
        F: Function to be calculated. Assume F is 
        initialized as F(Nx,Ny,Nz), where Nx, Ny, Nz
        is coordinate of axis-x, axis-y, and axis-z.
        
        dx,dy,dz: difference of x, y, and z.
        
    Central difference is applied to interior 
    points. For boundary points,
    
    -5u[i+1] + 4u[i+2] - u[i+3] + 2u[i], i = 0
    -5u[i-1] + 4u[i-2] - u[i-3] + 2u[i], i = N
    
    This code is ported from hand-written matlab funcion 
    using vectorization method. The points on the 
    boundary are dealt with separately. First, the points
    on surface are calculated, then the edge, finally the 
    vertexs.
    
    last revised on 2022/3/4
    
    !!
    https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    this article can save me a week!!! 2022/3/25
    """
    
    F = np.array(F)
    assert all(np.array(F.shape) >= 4),"At least one dimension is less than 4"
    [Ny, Nx, Nz] = F.shape
    
    laplacian = np.zeros(F.shape)
    xi = np.arange(1,Nx-1)
    yi = np.arange(1,Ny-1)
    zi = np.arange(1,Nz-1)
    xi = xi.reshape(1,Nx-2,1)  
    yi = yi.reshape(Ny-2,1,1)
    zi = zi.reshape(1,1,Nz-2)


    # interior points
    laplacian[yi,xi,zi] = ((1/dx**2)*(F[yi,xi+1,zi] - 2*F[yi,xi,zi] + F[yi,xi-1,zi]) +
        (1/dy**2)*(F[yi+1,xi,zi] - 2*F[yi,xi,zi] + F[yi-1,xi,zi]) +
        (1/dz**2)*(F[yi,xi,zi+1] - 2*F[yi,xi,zi] + F[yi,xi,zi-1]))

    
    # boundary index array(with shape corresponding to each dimension)
    cutx = np.array([0,-1]).reshape(1,2,1) # boundary points for axis-x
    cuty = np.array([0,-1]).reshape(2,1,1) # boundary points for axis-y
    cutz = np.array([0,-1]).reshape(1,1,2) # boundary points for axis-z
   
   
    # boundary plane(exclude edges)
    # xy-plane along x and y
    laplacian[yi,xi,[0,-1]] = ((1/dx**2)*(F[yi,xi+1,[0,-1]] - 2*F[yi,xi,[0,-1]] + F[yi,xi-1,[0,-1]])  
        + (1/dy**2)*(F[yi+1,xi,[0,-1]] - 2*F[yi,xi,[0,-1]] + F[yi-1,xi,[0,-1]]) 
        + 0)
    laplacian[yi,xi,[0,-1]] = (np.concatenate((
        laplacian[yi,xi,0] + (1/dz**2)*(-5*F[yi,xi,1] + 4*F[yi,xi,2] - F[yi,xi,3] + 2*F[yi,xi,0]), 
        laplacian[yi,xi,-1] + (1/dz**2)*(-5*F[yi,xi,-1-1] + 4*F[yi,xi,-1-2] - F[yi,xi,-1-3] + 2*F[yi,xi,-1])),
        axis=2)) # along z

    # xz-plane along x and z
    laplacian[cuty,xi,zi] = ((1/dx**2)*(F[cuty,xi+1,zi] - 2*F[cuty,xi,zi] + F[cuty,xi-1,zi]) 
        + 0 
        + (1/dz**2)*(F[cuty,xi,zi+1] - 2*F[cuty,xi,zi] + F[cuty,xi,zi-1]))
    laplacian[cuty,xi,zi] = (np.concatenate((
        laplacian[0,xi,zi] + (1/dy**2)*(-5*F[1,xi,zi] + 4*F[2,xi,zi] - F[3,xi,zi] + 2*F[0,xi,zi]), 
        laplacian[-1,xi,zi] + (1/dy**2)*(-5*F[-1-1,xi,zi] + 4*F[-1-2,xi,zi] - F[-1-3,xi,zi] + 2*F[-1,xi,zi])),
        axis=0)) # along y

    # yz-plane along y and z
    laplacian[yi,cutx,zi] = ( 0
        + (1/dy**2)*(F[yi+1,cutx,zi] - 2*F[yi,cutx,zi] + F[yi-1,cutx,zi])  
        + (1/dz**2)*(F[yi,cutx,zi+1] - 2*F[yi,cutx,zi] + F[yi,cutx,zi-1]))
    laplacian[yi,cutx,zi] = (np.concatenate(( 
        laplacian[yi,0,zi] + (1/dx**2)*(-5*F[yi,1,zi] + 4*F[yi,2,zi] - F[yi,3,zi] + 2*F[yi,0,zi]),  
        laplacian[yi,-1,zi] + (1/dx**2)*(-5*F[yi,-1-1,zi] + 4*F[yi,-1-2,zi] - F[yi,-1-3,zi] + 2*F[yi,-1,zi])),
        axis=1)) # along x


    # edges(exclude vertexs)
    # x-line
    bd_x_lines = (1/dx**2)*(F[cuty,xi+1,cutz] - 2*F[cuty,xi,cutz] + F[cuty,xi-1,cutz])
    x_yl_arg1 = (1/dy**2)*(-5*F[1,xi,[0,-1]] + 4*F[2,xi,[0,-1]] - F[3,xi,[0,-1]] + 2*F[0,xi,[0,-1]])
    x_yl_arg2 = (1/dy**2)*(-5*F[-1-1,xi,[0,-1]] + 4*F[-1-2,xi,[0,-1]] - F[-1-3,xi,[0,-1]] + 2*F[-1,xi,[0,-1]])
    bd_x_lines = (np.concatenate((bd_x_lines[0,:,:]  + x_yl_arg1,
                                  bd_x_lines[1,:,:] + x_yl_arg2), axis=0 ))  # along y
    x_zl_arg1 = (1/dz**2)*(-5*F[[0,-1],xi,1] + 4*F[[0,-1],xi,2] - F[[0,-1],xi,3] + 2*F[[0,-1],xi,0])
    x_zl_arg2 = (1/dz**2)*(-5*F[[0,-1],xi,-1-1] + 4*F[[0,-1],xi,-1-2] - F[[0,-1],xi,-1-3] + 2*F[[0,-1],xi,-1])
    x_zl_arg1 = x_zl_arg1.reshape(2,Nx-2,1)
    x_zl_arg2 = x_zl_arg2.reshape(2,Nx-2,1)
    bd_x_lines = ( np.concatenate((bd_x_lines[:,:,0].reshape(2,Nx-2,1) + x_zl_arg1,
                                   bd_x_lines[:,:,1].reshape(2,Nx-2,1) + x_zl_arg2), axis=2))  # along z
    
    # y-line
    bd_y_lines = (1/dy**2)*(F[yi+1,cutx,cutz] - 2*F[yi,cutx,cutz] + F[yi-1,cutx,cutz])
    y_xl_arg1 = (1/dx**2)*(-5*F[yi,1,[0,-1]] + 4*F[yi,2,[0,-1]] - F[yi,3,[0,-1]] + 2*F[yi,0,[0,-1]])
    y_xl_arg2 = (1/dx**2)*(-5*F[yi,-1-1,[0,-1]] + 4*F[yi,-1-2,[0,-1]] - F[yi,-1-3,[0,-1]] + 2*F[yi,-1,[0,-1]])
    y_xl_arg1 = y_xl_arg1.reshape(Ny-2,1,2)
    y_xl_arg2 = y_xl_arg2.reshape(Ny-2,1,2)
    bd_y_lines = (np.concatenate((bd_y_lines[:,0,:].reshape(Ny-2,1,2) + y_xl_arg1,  
                                  bd_y_lines[:,1,:].reshape(Ny-2,1,2) + y_xl_arg2), axis=1)) # along x
    y_zl_arg1 = (1/dz**2)*(-5*F[yi,[0,-1],1] + 4*F[yi,[0,-1],2] - F[yi,[0,-1],3] + 2*F[yi,[0,-1],0])
    y_zl_arg2 = (1/dz**2)*(-5*F[yi,[0,-1],-1-1 ] + 4*F[yi,[0,-1],-1-2] - F[yi,[0,-1],-1-3] + 2*F[yi,[0,-1],-1])
    y_zl_arg1 = y_zl_arg1.reshape(Ny-2,2,1)
    y_zl_arg2 = y_zl_arg2.reshape(Ny-2,2,1)
    bd_y_lines = (np.concatenate((bd_y_lines[:,:,0].reshape(Ny-2,2,1) + y_zl_arg1,
                                  bd_y_lines[:,:,1].reshape(Ny-2,2,1) + y_zl_arg2), axis=2))  # along z 
    
    # z-line
    bd_z_lines = (1/dz**2)*(F[cuty,cutx,zi+1] - 2*F[cuty,cutx,zi] + F[cuty,cutx,zi-1])
    z_yl_arg1 = (1/dy**2)*(-5*F[1,cutx,zi] + 4*F[2,cutx,zi] - F[3,cutx,zi] + 2*F[0,cutx,zi])
    z_yl_arg2 = (1/dy**2)*(-5*F[-1-1,cutx,zi] + 4*F[-1-2,cutx,zi] - F[-1-3,cutx,zi] + 2*F[-1,cutx,zi])
    z_yl_arg1 = z_yl_arg1.reshape(1,2,Nz-2)
    z_yl_arg2 = z_yl_arg2.reshape(1,2,Nz-2)
    bd_z_lines = (np.concatenate((bd_z_lines[0,:,:].reshape(1,2,Nz-2) + z_yl_arg1,
                                  bd_z_lines[1,:,:].reshape(1,2,Nz-2) + z_yl_arg2), axis=0)) # along y

    z_xl_arg1 = (1/dx**2)*(-5*F[cuty,1,zi] + 4*F[cuty,2,zi] - F[cuty,3,zi] + 2*F[cuty,0,zi])
    z_xl_arg2 = (1/dx**2)*(-5*F[cuty,-1-1,zi] + 4*F[cuty,-1-2,zi] - F[cuty,-1-3,zi] + 2*F[cuty,-1,zi])
    bd_z_lines = (np.concatenate((bd_z_lines[:,0,:].reshape(2,1,Nz-2) + z_xl_arg1,   
                                  bd_z_lines[:,1,:].reshape(2,1,Nz-2) + z_xl_arg2),axis=1)) # along x


    laplacian[cuty,xi,cutz] = bd_x_lines
    laplacian[yi,cutx,cutz]= bd_y_lines
    laplacian[cuty,cutx,zi] = bd_z_lines


    # vertexs(8 vertexs in total)
    leftdown = ((1/dx**2)*(-5*F[0,1,cutz] + 4*F[0,2,cutz] - F[0,3,cutz] + 2*F[0,0,cutz]) 
        + (1/dy**2)*(-5*F[1,0,cutz] + 4*F[2,0,cutz] - F[3,0,cutz] + 2*F[0,0,cutz]))
    leftup = ((1/dx**2)*(-5*F[0,1,cutz] + 4*F[0,2,cutz] - F[0,3,cutz] + 2*F[0,0,cutz]) 
        + (1/dy**2)*(-5*F[-1-1,0,cutz] + 4*F[-1-2,0,cutz] - F[-1-3,0,cutz] + 2*F[-1,0,cutz]))
    rightup = ((1/dx**2)*(-5*F[0,-1-1,cutz] + 4*F[0,-1-2,cutz] - F[0,-1-3,cutz] + 2*F[0,-1,cutz]) 
        + (1/dy**2)*(-5*F[-1-1,0,cutz] + 4*F[-1-2,0,cutz] - F[-1-3,0,cutz] + 2*F[-1,0,cutz]))
    rightdown = ((1/dx**2)*(-5*F[0,-1-1,cutz] + 4*F[0,-1-2,cutz] - F[0,-1-3,cutz] + 2*F[0,-1,cutz]) 
        + (1/dy**2)*(-5*F[1,0,cutz] + 4*F[2,0,cutz] - F[3,0,cutz] + 2*F[0,0,cutz]))

    # TODO: equivalent to np.block??
    bd_points = np.array([[leftdown, rightdown],[leftup, rightup]]).reshape(2,2,2)
    bd_p_arg1 = (1/dz**2)*(-5*F[cuty,cutx,1] + 4*F[cuty,cutx,2] - F[cuty,cutx,3] + 2*F[cuty,cutx,0])
    bd_p_arg2 = (1/dz**2)*(-5*F[cuty,cutx,-1-1] + 4*F[cuty,cutx,-1-2] - F[cuty,cutx,-1-3] + 2*F[cuty,cutx,-1])
    bd_p_arg1 = bd_p_arg1.reshape(2,2,1)
    bd_p_arg2 = bd_p_arg2.reshape(2,2,1)
    bd_points[:,:,[0,-1]] = (np.concatenate((bd_points[:,:,0].reshape(2,2,1) + bd_p_arg1, 
                                             bd_points[:,:,-1].reshape(2,2,1) + bd_p_arg2), axis=2)) # along z


    laplacian[cuty,cutx,cutz] = bd_points

    return laplacian







def laplacian(dx, dy, dz, w):
    """ Calculate the laplacian of the array w=[] """
    laplacian = np.zeros(w.shape)
    for z in range(1,w.shape[2]-1):
        laplacian[:, :, z] = (1/dz)**2 * ( w[:, :, z+1] - 2*w[:, :, z] + w[:, :, z-1] )
    for y in range(1,w.shape[0]-1):
        laplacian[y, ...] = laplacian[y, :, :] + (1/dy)**2 * ( w[y+1, :, :] - 2*w[y, :, :] + w[y-1, :, :] )
    for x in range(1,w.shape[1]-1):
        laplacian[:, x,:] = laplacian[:, x, :] + (1/dx)**2 * ( w[:, x+1, :] - 2*w[:, x, :] + w[:, x-1, :] )
    return laplacian


if __name__ == "__main__":
    Lx = 2
    Ly = 2
    Lz = 2
    Nx = 40
    Nz = 40
    Ny = 40
    x = np.linspace(-Lx,Lx,Nx)
    y = np.linspace(-Ly,Ly,Ny)
    z = np.linspace(-Lz,Lz,Nz)
    dx = 2*Lx/(Nx-1)
    dy = 2*Ly/(Ny-1)
    dz = 2*Lz/(Nz-1)

    [X,Y,Z] = np.meshgrid(x,y,z)

    F1 = np.sin(X)
    F2 = X**2 + Y**2 + Z**2
    F3 = (np.cos(X) + np.sin(Y))**2 + Y**2  + Z**2
    F4 = np.exp(X+Y**2)/(np.log10(abs(Z))+1)

    del2_F1 = vectorized(F1,dx,dy,dz)
    del2_F2 = vectorized(F2,dx,dy,dz)
    del2_F3 = vectorized(F3,dx,dy,dz)
    del2_F4 = vectorized(F4,dx,dy,dz)

    py_del2_F1 = laplace(F1)
    py_del2_F2 = laplace(F2)
    py_del2_F3 = laplace(F3,mode='nearest')
    py_del2_F4 = laplace(F4)
    F1_sol = -np.sin(Z)
    F3_sol = -2*( 2*np.cos(X)*np.sin(Y)+np.cos(2*X+np.sin(Y)**2 - np.cos(Y)**2-1 ))