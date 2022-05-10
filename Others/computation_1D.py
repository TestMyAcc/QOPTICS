# %%
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
# %%
# Meta-parameters

Nx = 1000
Lx = 5
x = np.linspace(-Lx,Lx,Nx)
pi = 3.14159265359
dx = np.diff(x)[0]
dw = 1e-6   # condition for converge : <1e-3*dx**2        

# Some constants
hbar = 2.054571800139113e-34 
m = 2.411000000000000e-25 # Rb atoms
# BEC parameters
As = 6.82e-09
Nbec = 10000
Wx = 1
unit = np.sqrt(hbar/(m*Wx))
G = (4*pi*hbar**2*As*Nbec/m)*unit**-3*(hbar*Wx)**-1

Epot = 0.5*x**2
mu = (9/32)**(1/3) * G**(2/3)
TF_amp = (mu-Epot)/G
np.clip(TF_amp, 0, np.inf,out=TF_amp)
TF_pbb = np.sqrt(TF_amp)
total = np.sum(np.abs(TF_pbb**2)*dx)
TFn_pbb = TF_pbb/np.sqrt(total)


# %% 
# Euler method
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

    
    for j in range(nj):
    
        _psiG = ( -dw*(  
            -0.5 * laplacian(_psiG,dx) +                      
            ( Epot + G*np.abs(_psiG)**2) * _psiG  
            - mu*_psiG ) + _psiG )
        

        if (j == 0 or j == nj-1):
            print(np.sum(np.abs(_psiG**2)*dx))
        # if (j % 100000) == 0:
        #  update energy constraint 
            
            # Nfactor = Normalize(_psiG,_psiE,dx,dy,dz)
            # J = J + 1 
            # psiGmu = psiGmu/(Nfactor)
            # psiEmu = psiEmu/(Nfactor)
            # psiGmuArray[J+1] = (np.sum(np.conjugate(_psiG)*  
            #     (- 0.5 * 2*_psiG*laplacian(_psiG,dx,dy,dz) +     #Operated _psiG
            #     (Epot + Ggg*np.abs(_psiG)**2 + Gge*np.abs(_psiE)**2) * _psiG )  
            #     *dx*dy*dz))
            # psiEmuArray[J+1] = (np.sum(np.conjugate(_psiE)*  
            #     (- 0.5 * 2*_psiE*laplacian(_psiE,dx,dy,dz) +     #Operated _psiE
            #     ( Epot  + Gee*np.abs(_psiE)**2 + Geg*np.abs(_psiG)**2)*_psiE) 
            #     *dx*dy*dz))
            
    return _psiG


#%%
@njit(fastmath=True, nogil=True)
def laplacian(w, _dx):
    """ Calculate the laplacian of the array w=[]."""
    laplacian = np.zeros(w.shape)
    for x in range(1,len(w)-1):
        laplacian[x] =  (1/_dx)**2 * ( w[x+1] - 2*w[x] + w[x-1])
    return laplacian


# %%
nj = 50000000
psiG = compute_BEC_Euler(TFn_pbb, nj)


# %%
