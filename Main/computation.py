# %% 
# BEGINNING(CPU calculation)
from numba import njit
import numpy as np
import h5py
from scipy.special import genlaguerre
import os

# %%
# Meta-parameters

Nx = 121
Ny = 121
Nz = 121
Lx = 10
Ly = 10
Lz = 10
x = np.linspace(-Lx,Lx,Nx)
y = np.linspace(-Ly,Ly,Ny)
z = np.linspace(-Lz,Lz,Nz)
Nx,Ny,Nz = len(x), len(y), len(z)
[X,Y,Z] = np.meshgrid(x,y,z)
dx = np.diff(x)[0]
dy = np.diff(y)[0]
dz = np.diff(z)[0]
dw = 1e-6   # condition for converge : <1e-3*dx**2

# Some constants
pi = 3.14159265359
hbar = 1.054571800139113e-34 
m = 1.411000000000000e-25  # Rb atoms
# BEC parameters
As = 5.82e-09
Nbec = 10000
Rabi = 1000
Wx = 2000
Wy = 2000
Wz = 2000
# unit = 1.222614572474304e-06;
unit = np.sqrt(hbar/m/Wz)

Ggg = (4*pi*hbar**2*As*Nbec/m)*unit**-3*(hbar*Wz)**-1
Gee = Ggg  
Gge = 0
Geg = 0
Epot = ( (Wx**2*X**2 + Wy**2*Y**2 + Wz**2*Z**2 )
            / (2*Wz**2) )
Epot = np.array(Epot,dtype=np.cfloat)
psiGmu = (15*Ggg / ( 16*pi*np.sqrt(2) )  )**(2/5)  
psiEmu = (15*Gee / ( 16*pi*np.sqrt(2) )  )**(2/5) 

# psiGmu = (15*Ggg/(64*np.sqrt(2)*np.pi))**(2/5) # for oval potential
# %%
TF_amp = np.array((psiGmu-Epot)/Ggg)
np.clip(TF_amp, 0, np.inf,out=TF_amp)
TF_pbb = np.sqrt(TF_amp,dtype=np.cfloat)
total = np.sum(np.abs(TF_pbb)**2*dx*dy*dz)
n_TF_pbb = TF_pbb/np.sqrt(total)
n_TF_pbb = np.array(n_TF_pbb, dtype=np.cfloat)




def makeLG(X,Y,Z,_W0,_Lambda, _L):
    _P = 0
    Zrl = np.pi*_W0**2/_Lambda                         #Rayleigh length
    W= _W0*np.sqrt(1+(Z/Zrl)**2)  
    Rz = Z + Zrl**2/Z 
    Guoy = (abs(_L)+2*_P+1)*np.arctan2(Z,Zrl) 
    
    Nx = X.shape[1]
    Ny = Y.shape[0]
    Nz = Z.shape[2]
    LG = np.zeros((Nx,Ny,Nz), dtype=np.cfloat)
        
    R = np.sqrt(X**2 + Y**2)
    Phi = np.arctan2(Y,X)
    AL =((np.sqrt(2)*R/W))**abs(_L)
    ALpoly =genlaguerre(_P,abs(_L))(2*(R/W)**2)
    AGauss = np.exp(-(R/W)**2)
    Ptrans1 = np.exp(-1j*(2*np.pi/_Lambda)*R**2/(2*Rz)) # Here
    Ptrans2 = np.exp(-1j*_L*Phi)
    PGuoy = np.exp(1j*Guoy)
    LG = (_W0/W)*AL*ALpoly*AGauss*Ptrans1*Ptrans2*PGuoy

    if (_L == 0 and _P == 0):
        Plong = np.exp(-1j*((2*np.pi/_Lambda)*Z - Guoy))
        LG = (_W0/W)*AGauss*Ptrans1*Ptrans2*Plong
    
    LG = 1*LG/np.max(np.abs(LG)) 
    
    return LG


@njit(fastmath=True, nogil=True)
def compute_BEC_Euler(LG:np.ndarray, nj, stepJ):
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

    psiG = np.asarray(n_TF_pbb*0.1)
    psiE = np.zeros_like(n_TF_pbb)
    
    # print("abs|n_TF_pbb|^2")
    # print(np.sum(np.abs(n_TF_pbb)**2*dx*dy*dz))
    # print("abs|psiG|^2")
    # print(np.sum(np.abs(psiG)**2*dx*dy*dz))
    # print("abs|psiE|^2")
    # print(np.sum(np.abs(psiE)**2*dx*dy*dz))
    
    
    
    psiG_n = np.zeros_like(psiG)
    
    for j in range(nj):
        psiG_n = dw * (0.5 * laplacian(psiG,dx,dy,dz ) - (Epot + Ggg*np.abs(psiG)**2 + Gge*np.abs(psiE)**2) * psiG \
                    - np.conjugate(LG)*psiE + psiGmu*psiG) \
                + psiG 
        psiE = dw * (0.5 * laplacian(psiE,dx,dy,dz) - (Epot + Gee*np.abs(psiE)**2 + Geg*np.abs(psiG)**2) * psiE \
                    - LG*psiG  + psiEmu*psiE)  \
                + psiE
                
        psiG = psiG_n

        if np.mod(j, stepJ) == 0 and j != 0:
            print(np.sum(np.abs(psiG)**2*dx*dy*dz))
            print(np.sum(np.abs(psiE)**2*dx*dy*dz))
    return psiG, psiE



@njit(fastmath=True, nogil=True)
def laplacian(w, dx,dy,dz):
    lap = np.zeros_like(w)
    for i in range(1,w.shape[0]-1):
        for j in range(1, w.shape[1]-1):
            for k in range(1, w.shape[2]-1):
                lap[i,j,k] = (1/dx)**2 * ( w[i,j+1,k] - 2*w[i,j,k] + w[i,j-1,k]) + \
                            (1/dy)**2 * ( w[i+1,j,k] - 2*w[i,j,k] + w[i-1,j,k]) + \
                            (1/dz)**2 * ( w[i,j,k+1] - 2*w[i,j,k] + w[i,j,k-1])
                    
    return lap



@njit(fastmath=True, nogil=True)
def Normalize(_psiG,_psiE,_dx,_dy,_dz):
    SumPsiG = np.sum( np.abs(_psiG)**2*_dx*_dy*_dz)
    SumPsiE = np.sum( np.abs(_psiE)**2*_dx*_dy*_dz)
    Nfactor = SumPsiG +  SumPsiE  
    return Nfactor

# %%
nj = 10000
stepJ = nj

base = os.path.join(os.path.expanduser("~"),"Data")
print("\n Total runs {} steps , update every {} steps\n".format(nj, stepJ))
for i in range(1,2):
    LG = makeLG(X,Y,Z,1,1, i)
    psiG, psiE = compute_BEC_Euler(LG,nj,stepJ)
    filename = f"scan_param_sequence_cpu_L{i}"
    path = os.path.join(base, filename) + '.h5'
    with h5py.File(path, "w") as f:
        f['psiG'] = psiG
        f['psiE'] = psiE
        f['LG'] = LG
        f['L'] = i
        f['Metaparameters/Nx'] = Nx
        f['Metaparameters/Ny'] = Ny
        f['Metaparameters/Nz'] = Nz
        f['Metaparameters/Lx'] = Lx
        f['Metaparameters/Ly'] = Ly
        f['Metaparameters/Lz'] = Lz
        f['Metaparameters/dw'] = dw
        f['Parameters/As'] = As
        f['Parameters/Nbec'] = Nbec
        f['Parameters/Rabi'] = Rabi
        f['Parameters/m'] = m
        f['Parameters/Wx'] = Wx
        f['Parameters/Wy'] = Wy
        f['Parameters/Wz'] = Wz
        f['Parameters/dw'] = dw
        f['Parameters/Ggg'] = Ggg
        f['Parameters/Gee'] = Gee
        f['Parameters/Gge'] = Gge
        f['Parameters/Geg'] = Geg
        print(f"\nstore{filename}\n")

# %%
