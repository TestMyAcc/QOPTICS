# %%
#BEGINNING
from fileinput import filename
from this import d
from numba import njit
import cupy as cp
import numpy as np
import h5py
# %%
# Meta-parameters

Nx = 121
Ny = 121
Nz = 121
Lx = 60
Ly = 60
Lz = 60
x = np.linspace(-Lx,Lx,Nx)
y = np.linspace(-Ly,Ly,Ny)
z = np.linspace(-Lz,Lz,Nz)
Nx,Ny,Nz = len(x), len(y), len(z)
[grid_x,grid_y,grid_z] = np.meshgrid(x,y,z)
pi = 3.14159265359
dx = np.diff(x)[0]
dy = np.diff(y)[0]
dz = np.diff(z)[0]
dw = 1e-6   # condition for converg<1e-3*dx**2

# Some constants
hbar = 1.054571800139113e-34 
m = 1.411000000000000e-25  # Rb atoms
# BEC parameters
As = 5.82e-09
Nbec = 10000
Rabi = 1000
Wx = 1.38e-23/hbar
Wy = 1.38e-23/hbar
Wz = 1.38e-23/hbar
# unit = 1.222614572474304e-06;
unit = np.sqrt(hbar/m/Wz)

Ggg = (4*pi*hbar**2*As*Nbec/m)*unit**-3*(hbar*Wz)**-1
Gee = Ggg  
Gge = 0
Geg = 0
Epot = ( (Wx**2*grid_x**2 + Wy**2*grid_y**2 + Wz**2*grid_z**2 )
            / (2*Wz**2) )
psiGmu = (15*Ggg / ( 16*pi*np.sqrt(2) )  )**(2/5)  
psiEmu = (15*Gee / ( 16*pi*np.sqrt(2) )  )**(2/5) # for circular potential

# psiGmu = (16*Ggg/(64*np.sqrt(2)*np.pi))**(2/5) # for oval potential
# %%
TF_amp = np.array((psiGmu-Epot)/Ggg,dtype=np.cfloat)
np.clip(TF_amp, 0, np.inf,out=TF_amp)
TF_pbb = np.sqrt(TF_amp)
total = np.sum(np.abs(TF_pbb)**2*dx*dy*dz)
n_TF_pbb = TF_pbb/np.sqrt(total)
# Laguerre-Gaussian laser
#%%
import os
base_dir = r'C:\\Users\\Lab\\Desktop\\Data\\local\\'
print(f"reading LG data below {base_dir}\n")
# filename = input("reading LG data from which file(subfolder/file)\n: ")
filename = 'LG10_121-121-121'
lgpath = os.path.join(base_dir, filename) + '.h5'

if (os.path.exists(lgpath)):
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



# %%

def compute_BEC_Euler(Epot:np.ndarray, psiG:np.ndarray, psiE:np.ndarray, LG:np.ndarray, nj:int):
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

    _psiG = cp.array(psiG, dtype=cp.complex128)
    _psiG_n = cp.zeros_like(_psiG)
    
    _psiE = cp.array(psiE,dtype=cp.complex128)
    
    _LG = cp.array(LG, dtype=cp.complex128)
    _Epot = cp.array(Epot, dtype=cp.complex128)
    
    _Lap = cp.zeros_like(psiG)
    
    for _ in range(nj):
        _Lap[1:Ny-1,1:Nx-1,1:Nz-1] = (
                (0.5/dx**2)*(
                        _psiG[2:Ny,   1:Nx-1, 1:Nz-1] 
                    - 2*_psiG[1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + _psiG[0:Ny-2, 1:Nx-1, 1:Nz-1])
                +(0.5/dy**2)*(
                        _psiG[1:Ny-1, 2:Nx,   1:Nz-1] 
                    - 2*_psiG[1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + _psiG[1:Ny-1, 0:Nx-2, 1:Nz-1])
                +(0.5/dz**2)*(
                        _psiG[1:Ny-1, 1:Nx-1, 2:Nz]
                    - 2*_psiG[1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + _psiG[1:Ny-1, 1:Nx-1, 0:Nz-2]))
        
        _psiG_n = dw * (_Lap - (_Epot + Ggg*cp.abs(_psiG)**2 + Gge*cp.abs(_psiE)**2) * _psiG \
                      - cp.conjugate(_LG)*_psiE + psiGmu*_psiG) + _psiG 
        # _psiG = dw * (-_Epot * _psiG + psiGmu*_psiG) + _psiG

        _Lap[1:Ny-1,1:Nx-1,1:Nz-1] = (
                (0.5/dx**2)*(
                        _psiE[2:Ny,   1:Nx-1, 1:Nz-1] 
                    - 2*_psiE[1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + _psiE[0:Ny-2, 1:Nx-1, 1:Nz-1])
                +(0.5/dy**2)*(
                        _psiE[1:Ny-1, 2:Nx,   1:Nz-1] 
                    - 2*_psiE[1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + _psiE[1:Ny-1, 0:Nx-2, 1:Nz-1])
                +(0.5/dz**2)*(
                        _psiE[1:Ny-1, 1:Nx-1, 2:Nz]
                    - 2*_psiE[1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + _psiE[1:Ny-1, 1:Nx-1, 0:Nz-2]))
        _psiE = dw * ( _Lap - (_Epot + Gee*cp.abs(_psiE)**2 + Geg*cp.abs(_psiG)**2) * _psiE \
                      - _LG*_psiG  + psiEmu*_psiE) + _psiE
        # _psiE = dw * (-_Epot * _psiE + psiGmu*_psiE) + _psiE
        
        _psiG = _psiG_n
        
    return _psiG.get(), _psiE.get()



# %%
import matplotlib.pyplot as plt
nj = 100
stepJ = 2000
psiG = np.array(n_TF_pbb+0.1,dtype=np.cfloat)
psiE = np.zeros_like(n_TF_pbb,dtype=np.cfloat)
# psiG = np.array(np.ones(TF_pbb.shape)+5,dtype=np.cfloat)
# psiE = np.array(np.ones(TF_pbb.shape)+5,dtype=np.cfloat)
# %%
print(np.sum(np.abs(TF_pbb)**2*dx*dy*dz))
print(np.sum(np.abs(psiG)**2*dx*dy*dz))
print(np.sum(np.abs(psiE)**2*dx*dy*dz))
#%%
plt.figure()
plt.plot(x, np.abs(n_TF_pbb[61,:,61])**2*dx*dy*dz)
plt.figure()
plt.plot(y, np.abs(n_TF_pbb[:,61,61]**2)*dx*dy*dz)
plt.figure()
plt.plot(z, np.abs(n_TF_pbb[61,61,:]**2)*dx*dy*dz)
plt.figure()
plt.plot(x, np.abs(psiG[61,:,61])**2*dx*dy*dz)
plt.figure()
plt.plot(y, np.abs(psiG[:,61,61]**2)*dx*dy*dz)
plt.figure()
plt.plot(z, np.abs(psiG[61,61,:]**2)*dx*dy*dz)
plt.figure()
plt.plot(x, np.abs(psiE[61,:,61])**2*dx*dy*dz)
plt.figure()
plt.plot(y, np.abs(psiE[:,61,61]**2)*dx*dy*dz)
plt.figure()
plt.plot(z, np.abs(psiE[61,61,:]**2)*dx*dy*dz)
# print("\n Total runs {} steps , update every {} steps\n".format(nj, stepJ))
#%%
psiG, psiE = compute_BEC_Euler(Epot, psiG, psiE,LG,nj)
# %%

