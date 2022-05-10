#%% compare execution time
import cupyx.scipy as cuxsp
import numpy as np
import cupy as cp
import scipy as sp
import fdel2 #in unix
from numba import njit

def brutal(w, dx,dy,dz):
    lap = np.zeros_like(w)
    for i in range(1,w.shape[0]-1):
        for j in range(1, w.shape[1]-1):
            for k in range(1, w.shape[2]-1):
                lap[i,j,k] = (1/dx)**2 * ( w[i,j+1,k] - 2*w[i,j,k] + w[i,j-1,k]) + \
                            (1/dy)**2 * ( w[i+1,j,k] - 2*w[i,j,k] + w[i-1,j,k]) + \
                            (1/dz)**2 * ( w[i,j,k+1] - 2*w[i,j,k] + w[i,j,k-1])
                    
    return lap


#%%  for loop on CPU 
@njit(fastmath=True, nogil=True)
def loop(w, _dx, _dy, _dz):
    """ Calculate the del2 of the array w=[].
        Note that the boundary points aren't handled """
    lap = np.zeros_like(w)
    for z in range(1,w.shape[2]-1):
        lap[:, :, z] = (1/_dz)**2 * ( w[:, :, z+1] - 2*w[:, :, z] + w[:, :, z-1] )
    for y in range(1,w.shape[0]-1):
        lap[y, :,:] = lap[y, :, :] + (1/_dy)**2 * ( w[y+1, :, :] - 2*w[y, :, :] + w[y-1, :, :] )
    for x in range(1,w.shape[1]-1):
        lap[:, x,:] = lap[:, x, :] + (1/_dx)**2 * ( w[:, x+1, :] - 2*w[:, x, :] + w[:, x-1, :] )
    return lap


#%%  vectorized on CPU 
@njit(fastmath=True, nogil=True)
def vector_CPU(w, dx, dy, dz):
    Ny,Nx,Nz=w.shape
    # boundary is zero
    u=np.zeros_like(w)
    u[1:Ny-1,1:Nx-1,1:Nz-1] = (
        (1/dy**2)*(
                w[2:Ny,   1:Nx-1, 1:Nz-1] 
            - 2*w[1:Ny-1, 1:Nx-1, 1:Nz-1] 
            + w[0:Ny-2, 1:Nx-1, 1:Nz-1])
        +(1/dx**2)*(
                w[1:Ny-1, 2:Nx,   1:Nz-1] 
            - 2*w[1:Ny-1, 1:Nx-1, 1:Nz-1] 
            + w[1:Ny-1, 0:Nx-2, 1:Nz-1])
        +(1/dz**2)*(
                w[1:Ny-1, 1:Nx-1, 2:Nz]
            - 2*w[1:Ny-1, 1:Nx-1, 1:Nz-1] 
            + w[1:Ny-1, 1:Nx-1, 0:Nz-2]))
    return u

#%%   vectorized on GPU 
def vector_GPU(w, dx, dy, dz):
    w = cp.array(w)
    Ny,Nx,Nz=w.shape
    # boundary is zero
    u=cp.zeros_like(w)
    u[1:Ny-1,1:Nx-1,1:Nz-1] = (
        (1/dy**2)*(
                w[2:Ny,   1:Nx-1, 1:Nz-1] 
            - 2*w[1:Ny-1, 1:Nx-1, 1:Nz-1] 
            + w[0:Ny-2, 1:Nx-1, 1:Nz-1])
        +(1/dx**2)*(
                w[1:Ny-1, 2:Nx,   1:Nz-1] 
            - 2*w[1:Ny-1, 1:Nx-1, 1:Nz-1] 
            + w[1:Ny-1, 0:Nx-2, 1:Nz-1])
        +(1/dz**2)*(
                w[1:Ny-1, 1:Nx-1, 2:Nz]
            - 2*w[1:Ny-1, 1:Nx-1, 1:Nz-1] 
            + w[1:Ny-1, 1:Nx-1, 0:Nz-2]))
    return u.get()

#%% einsum of dense matrix on CPU
def tensor_CPU(w,dx,dy,dz):
    N, _, _ = w.shape 
    D = np.eye(N,k=-1) + -2*np.eye(N,k=0) + np.eye(N,k=1)
    D[0,:] = D[-1,:] = np.zeros((1,Nx))
    lap = np.zeros_like(w)
    lap = (1/dy**2) * np.einsum('ij,jkl->ikl', D, w) + \
          (1/dx**2) * np.einsum('ij,kjl->kil', D, w) + \
          (1/dz**2) * np.einsum('ij,klj->kli', D, w)
    return lap
#%% einsum of dense matrix on GPU
def tensor_GPU(w,dx,dy,dz):
    w = cp.array(w)
    N, _, _ = w.shape 
    D = cp.eye(N,k=-1) + -2*cp.eye(N,k=0) + cp.eye(N,k=1)
    D[0,:] = D[-1,:] = cp.zeros((1,Nx))
    lap = cp.zeros_like(w)
    lap = (1/dy**2) * cp.einsum('ij,jkl->ikl', D, w) + \
          (1/dx**2) * cp.einsum('ij,kjl->kil', D, w) + \
          (1/dz**2) * cp.einsum('ij,klj->kli', D, w)
    return lap.get()
#%% sparse matrix on CPU
def sparse_CPU(w,dx,dy,dz):
    N, _, _ = w.shape 
    ex = np.ones(N)
    values = np.repeat(ex,3).reshape(3,N)
    values[1,:] *= -2
    values[1,0] = values[1,-1]  = 0  
    values[0,-2] = values[2,1] =  0
    #cannot import name 'dia_array' from 'scipy.sparse' (C:\Users\Lab\anaconda3\envs\GPU\lib\site-packages\scipy\sparse\__init__.py)
    D = sp.sparse.dia_matrix((values,[-1,0,1]),shape=(N,N))
    lap = np.zeros_like(w)
    for i in range(1,N-1): 
        lap[:, :, i] += (1/dy)**2* (D @ w[:,:,i]) + (1/dx)**2 * (D @ w[:,:,i].T).T
        lap[i, :, :] += (1/dz)**2* (D @ w[i,:,:].T).T

    return lap
    
    
#%% sparse matrix on GPU
def sparse_GPU(w,dx,dy,dz):
    w = cp.array(w)
    N,_,_ = w.shape
    ex = cp.ones(N)
    values = cp.repeat(ex,3).reshape(3,N)
    values[1,:] *= -2
    values[1,0] = values[1,-1]  = 0  
    values[0,-2] = values[2,0] =  0
    D = cuxsp.sparse.diags(values,[-1,0,1],shape=(N,N))
    lap = cp.zeros_like(w)
    for i in range(1,N-1): 
        lap[:, :, i] += (1/dy)**2* (D @ w[:,:,i]) + (1/dx)**2 * ( D @ w[:,:,i].T ).T
        lap[i, :, :] += (1/dz)**2* (D @ w[i,:,:].T).T

    return lap.get()
   
   
#%%
Nx = 80
Ny = 80
Nz = 80
Lx = 0.1
Ly = 0.2
Lz = 0.3
x = np.linspace(-Lx,Lx,Nx)
y = np.linspace(-Ly,Ly,Ny)
z = np.linspace(-Lz,Lz,Nz)
dx = np.diff(x)[0]
dy = np.diff(y)[0]
dz = np.diff(z)[0]
[X,Y,Z] = np.meshgrid(x,y,z)

F1 = np.cos(X) + np.sin(Y) + Z**3
F2 = np.sin(Z)
F3 = X**2 + Y**2 + Z**2
F4 = (np.cos(X) + np.sin(Y))**2 + Y**2  + Z**2

sol1 = -np.cos(X) - np.sin(Y) + 6*Z
sol2 = -np.sin(Z)
sol3 = np.ones(F3.shape)*6
sol4 = -2*( 2*np.cos(X)*np.sin(Y)+np.cos(2*X)+np.sin(Y)**2 - np.cos(Y)**2-2)
l1 = loop(F4,dx,dy,dz)
l2 = vector_CPU(F4,dx,dy,dz)
l3 = tensor_CPU(F4,dx,dy,dz)
l4 = sparse_CPU(F4,dx,dy,dz)
l5 = vector_GPU(F4,dx,dy,dz)
l6 = tensor_GPU(F4,dx,dy,dz)
l7 = sparse_GPU(F4,dx,dy,dz)
l8 = brutal(F4,dx,dy,dz)
#%%
# print(l1[:,:,cut],end='\n=====\n')
print(np.allclose(l1[1:-1,1:-1,1:-1],sol4[1:-1,1:-1,1:-1],atol=1e-3))
# print(l2[:,:,cut],end='\n=====\n')
print(np.allclose(l2[1:-1,1:-1,1:-1],sol4[1:-1,1:-1,1:-1],atol=1e-3))
# print(l3[:,:,cut],end='\n=====\n')
print(np.allclose(l3[1:-1,1:-1,1:-1],sol4[1:-1,1:-1,1:-1],atol=1e-3))
# print(l4[:,:,cut],end='\n=====\n')
print(np.allclose(l4[1:-1,1:-1,1:-1],sol4[1:-1,1:-1,1:-1],atol=1e-3))
# print(l5[:,:,cut],end='\n=====\n')
print(np.allclose(l5[1:-1,1:-1,1:-1],sol4[1:-1,1:-1,1:-1],atol=1e-3))
# print(l6[:,:,cut],end='\n=====\n')
print(np.allclose(l6[1:-1,1:-1,1:-1],sol4[1:-1,1:-1,1:-1],atol=1e-3))
# print(l7[:,:,cut],end='\n=====\n')
print(np.allclose(l7[1:-1,1:-1,1:-1],sol4[1:-1,1:-1,1:-1],atol=1e-3))
print(np.allclose(l8[1:-1,1:-1,1:-1],sol4[1:-1,1:-1,1:-1],atol=1e-3))
#%%
l8 = fdel2.del2_loop(F1,dx,dy,dz)
l9 = fdel2.del2_ele(F1,dx,dy,dz)
print(np.allclose(l8[1:-1,1:-1,1:-1],sol1[1:-1,1:-1,1:-1],atol=1e-3))
print(np.allclose(l9[1:-1,1:-1,1:-1],sol1[1:-1,1:-1,1:-1],atol=1e-3))
# %%
