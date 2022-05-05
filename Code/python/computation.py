# %% 
# BEGINNING(CPU calculation)
from numba import njit
import numpy as np
import h5py
import dirutils,os
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
[grid_x,grid_y,grid_z] = np.meshgrid(x,y,z)
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
Epot = ( (Wx**2*grid_x**2 + Wy**2*grid_y**2 + Wz**2*grid_z**2 )
            / (2*Wz**2) )

psiGmu = (15*Ggg / ( 16*pi*np.sqrt(2) )  )**(2/5)  
psiEmu = (15*Gee / ( 16*pi*np.sqrt(2) )  )**(2/5) 

# psiGmu = (15*Ggg/(64*np.sqrt(2)*np.pi))**(2/5) # for oval potential
# %%
TF_amp = np.array((psiGmu-Epot)/Ggg,dtype=np.cfloat)
np.clip(TF_amp, 0, np.inf,out=TF_amp)
TF_pbb = np.sqrt(TF_amp)
total = np.sum(np.abs(TF_pbb)**2*dx*dy*dz)
n_TF_pbb = TF_pbb/np.sqrt(total)

#%%
# Laguerre-Gaussian laser
lgfilename = input("Specify filename\n"+dirutils.listLG()[1])
lgpath = os.path.join("~/Data/", lgfilename) + '.h5'
if (os.path.exists(os.path.expanduser(lgpath))):
    with h5py.File(os.path.expanduser(lgpath), "r") as f:
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
@njit(fastmath=True, nogil=True)
def compute_BEC_Euler(_psiG, _psiE, nj):
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

    _psiGmu = psiGmu
    _psiEmu = psiEmu
    _psiG_n = np.zeros_like(_psiG)
    
    for j in range(nj):
        
        
        _psiG_n = dw * (0.5 * laplacian(_psiG,dx,dy,dz ) - (Epot + Ggg*np.abs(_psiG)**2 + Gge*np.abs(_psiE)**2) * _psiG \
                      - np.conjugate(LG)*_psiE + _psiGmu*_psiG) \
                + _psiG 
        _psiE = dw * (0.5 * laplacian(_psiE,dx,dy,dz) - (Epot + Gee*np.abs(_psiE)**2 + Geg*np.abs(_psiG)**2) * _psiE \
                      - LG*_psiG  + _psiEmu*_psiE)  \
                + _psiE
                
        _psiG = _psiG_n

        if np.mod(j, stepJ) == 0:
            print(np.sum(np.abs(_psiG)**2*dx*dy*dz))
            print(np.sum(np.abs(_psiE)**2*dx*dy*dz))
    return _psiG, _psiE



@njit(fastmath=True, nogil=True)
def laplacian(w, dx, dy, dz):
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


# %%
@njit(fastmath=True, nogil=True)
def Normalize(_psiG,_psiE,_dx,_dy,_dz):
    SumPsiG = np.sum( np.abs(_psiG)**2*_dx*_dy*_dz)
    SumPsiE = np.sum( np.abs(_psiE)**2*_dx*_dy*_dz)
    Nfactor = SumPsiG +  SumPsiE  
    return Nfactor

# %%
import matplotlib.pyplot as plt
nj = 1000
stepJ = 50
psiGmuArray = np.zeros(int(nj/stepJ),dtype=np.float32)
psiEmuArray = np.zeros(int(nj/stepJ),dtype=np.float32)
psiGmuArray[0] = psiGmu
psiEmuArray[0] = psiEmu
psiG = np.array(n_TF_pbb+0.1,dtype=np.cfloat)
psiE = np.zeros_like(n_TF_pbb,dtype=np.cfloat)
# %%
print("abs|n_TF_pbb|^2")
print(np.sum(np.abs(n_TF_pbb)**2*dx*dy*dz))
print("abs|psiG|^2")
print(np.sum(np.abs(psiG)**2*dx*dy*dz))
print("abs|psiE|^2")
print(np.sum(np.abs(psiE)**2*dx*dy*dz))
#%%
plt.figure()
plt.plot(x, np.abs(n_TF_pbb[61,:,61])**2*dx*dy*dz)
plt.xlabel("x")
plt.title("n_TF_pbb")

plt.figure()
plt.plot(y, np.abs(n_TF_pbb[:,61,61]**2)*dx*dy*dz)
plt.xlabel("y")
plt.title("n_TF_pbb")

plt.figure()
plt.plot(z, np.abs(n_TF_pbb[61,61,:]**2)*dx*dy*dz)
plt.xlabel("z")
plt.title("n_TF_pbb")

plt.figure()
plt.plot(x, np.abs(psiG[61,:,61])**2*dx*dy*dz)
plt.xlabel("x")
plt.title("psiG")

plt.figure()
plt.plot(y, np.abs(psiG[:,61,61]**2)*dx*dy*dz)
plt.xlabel("y")
plt.title("psiG")

plt.figure()
plt.plot(z, np.abs(psiG[61,61,:]**2)*dx*dy*dz)
plt.xlabel("z")
plt.title("psiG")

plt.figure()
plt.plot(x, np.abs(psiE[61,:,61])**2*dx*dy*dz)
plt.xlabel("x")
plt.title("psiE")

plt.figure()
plt.plot(y, np.abs(psiE[:,61,61]**2)*dx*dy*dz)
plt.xlabel("y")
plt.title("psiE")

plt.figure()
plt.plot(z, np.abs(psiE[61,61,:]**2)*dx*dy*dz)
plt.xlabel("z")
plt.title("psiE")

plt.figure()
plt.plot(x, np.abs(LGdata[61,:,61])**2*dx*dy*dz)
plt.xlabel("x")
plt.title("LG")
print("\n Total runs {} steps , update every {} steps\n".format(nj, stepJ))
#%%
psiG, psiE, psiGmu, psiEmu = compute_BEC_Euler(psiG, psiE,nj)
# %%
# save data
import h5py
import os
import numpy as np
import dirutils

base_dir = os.path.join(os.path.expanduser("~"),"Data")
msg = f"storing data below {base_dir}"
filename = input(msg+'\n'+"Specify the filename: ")
path = (os.path.join(base_dir, filename)) + '.h5'

if os.path.exists(path) or filename == '':
    print("File already exists/Filename is empty!, do nothing.")    
else:
    with h5py.File(path, "w") as f:
        f['psiG'] = psiG
        f['psiE'] = psiE
        f['LGfile'] = lgpath
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
        print("storing succeeded!")


# %%
