# %%
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
# %%
# Meta-parameters

Nx = 61
Ny = 61
Lx = 20
Ly = 20
x = np.linspace(-Lx,Lx,Nx)
y = np.linspace(-Ly,Ly,Ny)
Nx,Ny = len(x), len(y)
[grid_x,grid_y] = np.meshgrid(x,y)
pi = 3.14159265359
dx = np.diff(x)[0]
dy = np.diff(y)[0]
dw = 1e-6   # condition for converge : <1e-3*dx**2

# Some constants
hbar = 1.054571800139113e-34 
m = 1.411000000000000e-25  # Rb atoms
# BEC parameters
Nbec = 10000
Rabi = 1000
Wx = 1.38e-23/hbar
Wy = 1.38e-23/hbar
unit = np.sqrt(hbar/m/Wx)

  
Epot = ( (Wx**2*grid_x**2 + Wy**2*grid_y**2)/(2*Wx**2) )
Mu = 100
TF_amp_up = Mu - Epot
np.clip(TF_amp_up, 0, np.inf,out=TF_amp_up)
G = np.sum(TF_amp_up*dx*dy)
TF_amp = TF_amp_up/G
TF_pbb = np.sqrt(TF_amp)
total = np.sum(np.abs(TF_pbb)**2*dx*dy)
n_TF_pbb = TF_pbb/np.sqrt(total)


# %%
@njit(fastmath=True, nogil=True)
def compute_BEC_Euler(_psiG, nj):
    """Calculating interaction between the BEC and L.G. beams.
    Two-order system is used. The code evaluates ground-
    state BEC and excited-state BEC, and save the data.
    Note: Data is calculated without units. Use Euler method
    to update time.
   
    Args:
        nj: Number of iterations.
        stepJ: Number of iterations to update energy constraint.
        isLight: interaction with light?.
        x,y,z: coordinate vectors
        dw: finite time difference.
    """

    _Mu = Mu
    
    for j in range(nj):
        
        _psiG = ( dw * (0.5 * laplacian(_psiG,dx,dy) 
                      - (Epot + G*np.abs(_psiG)**2) * _psiG
                + _Mu*_psiG) + _psiG)
        if np.mod(j, 2000) == 0:
            print(np.sum(np.abs(_psiG)**2*dx*dy))
        

    return _psiG

# %% 
@njit(fastmath=True, nogil=True)
def Hamiltonian(_psi, _G, _dx, _dy):
    Energy = (np.sum( (np.conjugate(_psi) *  
        (-0.5 *laplacian(_psi,dx,dy)+(Epot + _G*np.abs(_psi)**2)*_psi)*dx*dy)))

    return Energy

#%%
@njit(fastmath=True, nogil=True)
def laplacian(w, _dx, _dy):
    """ Calculate the laplacian of the array w=[].
        Note that the boundary points aren't handled """
    laplacian = np.zeros(w.shape)
    for y in range(1,w.shape[0]-1):
        laplacian[y, :] = laplacian[y, :] + (1/_dy)**2 * ( w[y+1, :] - 2*w[y, :] + w[y-1, :] )
    for x in range(1,w.shape[1]-1):
        laplacian[:, x] = laplacian[:, x] + (1/_dx)**2 * ( w[:, x+1] - 2*w[:, x] + w[:, x-1] )
    return laplacian


# %%
nj = 200000
# stepJ = 100000   # updating energy constraint
# print("\n Total runs {} steps , update every {} steps\n".format(nj, stepJ))
psiG = compute_BEC_Euler(n_TF_pbb+0.1,nj)
# %%
