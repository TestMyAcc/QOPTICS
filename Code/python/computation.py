# %% 
# BEGINNING(CPU calculation)
from numba import njit
import numpy as np
import h5py
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
pi = 3.14159265359
dx = np.diff(x)[0]
dy = np.diff(y)[0]
dz = np.diff(z)[0]
dw = 1e-6   # condition for converge : <1e-3*dx**2

# Some constants
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
# circular potential
psiGmu = (15*Ggg / ( 16*pi*np.sqrt(2) )  )**(2/5)  
psiEmu = (15*Gee / ( 16*pi*np.sqrt(2) )  )**(2/5) 

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
        
        
        _psiG_n = dw * (0.5 * del2(_psiG,dx,dy,dz ) - (Epot + Ggg*np.abs(_psiG)**2 + Gge*np.abs(_psiE)**2) * _psiG \
                      - np.conjugate(LG)*_psiE + _psiGmu*_psiG) \
                + _psiG 
        _psiE = dw * (0.5 * del2(_psiE,dx,dy,dz) - (Epot + Gee*np.abs(_psiE)**2 + Geg*np.abs(_psiG)**2) * _psiE \
                      - LG*_psiG  + _psiEmu*_psiE)  \
                + _psiE
                
        _psiG = _psiG_n

        # if (j % stepJ) == 0:
        # #  update energy constraint 
        #     Nfactor = Normalize(_psiG,_psiE,dx,dy,dz)
        #     # J = J + 1
        #     _psiGmu = _psiGmu/(Nfactor)
        #     _psiEmu = _psiEmu/(Nfactor)
            
            # psiEmuArray[J+1] = (np.sum(np.conjugate(_psiE)*  
            #     (- 0.5 * 2*_psiE*del2(_psiE,dx,dy,dz) +     #Operated _psiE
            #     ( Epot  + Gee*np.abs(_psiE)**2 + Geg*np.abs(_psiG)**2)*_psiE) 
            #     *dx*dy*dz))
        if np.mod(j, stepJ) == 0:
            print(np.sum(np.abs(_psiG)**2*dx*dy*dz))
            print(np.sum(np.abs(_psiE)**2*dx*dy*dz))
    return _psiG, _psiE

# %% 
@njit(fastmath=True, nogil=True)
def Hamiltonian(_psi, _G, _dx, _dy, _dz):
    Energy = (np.sum( (np.conjugate(_psi) *  
        (-0.5 *del2(_psi,dx,dy,dz)+(Epot + _G*np.abs(_psi)**2)*_psi)*dx*dy*dz)))

    return Energy

# %%
@njit(fastmath=True, nogil=True)
def compute_BEC_RK4(_psiG, _psiE, nj):
    """Calculating interaction between the BEC and L.G. beams.
    Two-order system is used. The code evaluates ground-
    state BEC and excited-state BEC, and save the data.
    Note: Data is calculated without units. Use RK4 method to update
    time.
   
    Args:
        nj: Number of iterations.
        stepJ: Number of iterations to update energy constraint.
        isLight: interaction with light?.
        x,y,z: coordinate vectors
        dw: finite time difference.
    """
    
    # Potential and initial condition
    _psiGmu = psiGmu
    # _psiEmu = psiEmu
    # _psiE = np.zeros(TFsol.shape)
    # _psiG = np.zeros(TFsol.shape)
    # _psiG = TFsol
    
    for j in range(nj):
        _psiG, _psiE = RK4(_psiG, _psiE, _psiGmu, _psiEmu, dw)
         
        if (j % stepJ) == 0:
        #  update energy constraint 
            Nfactor = Normalize(_psiG,_psiE,dx,dy,dz)
            _psiGmu = _psiGmu/(Nfactor)
            _psiEmu = _psiEmu/(Nfactor)
    
    return _psiG, _psiE

# %%
@njit(fastmath=True, nogil=True)
def G(_psiG, _psiE, _psiGmu):
    """Function of ground state"""
    tmp = np.zeros(_psiG.shape)
    tmp =  ( 0.5 * del2(_psiG,dx,dy,dz) 
         - (Epot + Ggg*np.abs(_psiG)**2 + Gge*np.abs(_psiE)**2)*_psiG
         + _psiGmu*_psiG )
    # set boundary points to zero
    tmp[0,:,:] = tmp[-1,:,:] = tmp[:,-1,:] = tmp[:,0,:] =  tmp[:,:,0] = tmp[:,:,-1] = 0
    return tmp
    
@njit(fastmath=True, nogil=True)
def E(_psiG, _psiE, _psiEmu):
    """Function of excited state"""
    tmp = np.zeros(_psiE.shape)
    tmp = ( 0.5 * del2(_psiE,dx,dy,dz) 
         - (Epot + Gee*np.abs(_psiE)**2 + Geg*np.abs(_psiG)**2)*_psiE
         + _psiEmu*_psiE )
    # set boundary points to zero
    tmp[0,:,:] =  tmp[-1,:,:] = tmp[:,-1,:] = tmp[:,0,:] =  tmp[:,:,0] = tmp[:,:,-1] = 0
    return tmp

@njit(fastmath=True, nogil=True)
def RK4(_psiG, _psiE, _psiGmu, _psiEmu, h):
    k_g = np.zeros((*_psiG.shape,4))
    k_e = np.zeros((*_psiG.shape,4))
    k_g[...,0] = G(_psiG, _psiE, _psiGmu) 
    k_e[...,0] = E(_psiG, _psiE, _psiEmu) 
    k_g[...,1] = G(_psiG + h*k_g[...,1]/2, _psiE + h*k_e[...,1]/2, _psiGmu) 
    k_e[...,1] = E(_psiG + h*k_g[...,1]/2, _psiE + h*k_e[...,1]/2, _psiEmu) 
    k_g[...,2] = G(_psiG + h*k_g[...,2]/2, _psiE + h*k_e[...,2]/2, _psiGmu) 
    k_e[...,2] = E(_psiG + h*k_g[...,2]/2, _psiE + h*k_e[...,2]/2, _psiEmu) 
    k_g[...,3] = G(_psiG + h*k_g[...,3], _psiE + h*k_e[...,3], _psiGmu) 
    k_e[...,3] = E(_psiG + h*k_g[...,3], _psiE + h*k_e[...,3], _psiEmu) 
    _psiG = _psiG + h/6 * ( k_g[...,0] + 2*k_g[...,1] + 2*k_g[...,2] + k_g[...,3] )
    _psiE = _psiE + h/6 * ( k_e[...,0] + 2*k_e[...,1] + 2*k_e[...,2] + k_e[...,3] )

    return _psiG, _psiE

@njit(fastmath=True, nogil=True)
def del2(w, _dx, _dy, _dz):
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

# @njit(fastmath=True, nogil=True)
def laplacian(w, dx, dy, dz):
    Ny,Nx,Nz=w.shape
    # u=np.zeros_like(w)
    w[1:Ny-1,1:Nx-1,1:Nz-1] = (
        (1/dx**2)*(
                w[2:Ny,   1:Nx-1, 1:Nz-1] 
            - 2*w[1:Ny-1, 1:Nx-1, 1:Nz-1] 
            + w[0:Ny-2, 1:Nx-1, 1:Nz-1])
        +(1/dy**2)*(
                w[1:Ny-1, 2:Nx,   1:Nz-1] 
            - 2*w[1:Ny-1, 1:Nx-1, 1:Nz-1] 
            + w[1:Ny-1, 0:Nx-2, 1:Nz-1])
        +(1/dz**2)*(
                w[1:Ny-1, 1:Nx-1, 2:Nz]
            - 2*w[1:Ny-1, 1:Nx-1, 1:Nz-1] 
            + w[1:Ny-1, 1:Nx-1, 0:Nz-2]))
    return w


# %%
@njit(fastmath=True, nogil=True)
def Normalize(_psiG,_psiE,_dx,_dy,_dz):
    SumPsiG = np.sum( np.abs(_psiG)**2*_dx*_dy*_dz)
    SumPsiE = np.sum( np.abs(_psiE)**2*_dx*_dy*_dz)
    Nfactor = SumPsiG +  SumPsiE  
    return Nfactor

# %%
import matplotlib.pyplot as plt
nj = 200
stepJ = 10
psiG = np.array(n_TF_pbb+0.1,dtype=np.cfloat)
psiE = np.zeros_like(n_TF_pbb,dtype=np.cfloat)
print(np.sum(np.abs(TF_pbb)**2*dx*dy*dz))
print(np.sum(np.abs(psiG)**2*dx*dy*dz))
print(np.sum(np.abs(psiE)**2*dx*dy*dz))
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
print("\n Total runs {} steps , update every {} steps\n".format(nj, stepJ))
#%%
psiG, psiE = compute_BEC_Euler(psiG, psiE,nj)






"""=============Main program above=============="""



# %%
# save data
import h5py
import os
import numpy as np


base_dir = r'c:\\Users\\Lab\\Desktop\\Data\\local\\'
print(f"storing data in {base_dir}....\n")
foldername = input("in which folder: ")
filename = input("the filename: ")
dirpath = os.path.join((os.path.join(base_dir, foldername)))
if os.path.isdir(dirpath) == False:
    print(f"create new dir {dir}....\n")
    os.mkdir(dirpath)
path = os.path.join(dirpath, filename) + '.h5'
if os.path.exists(path):
    print("File already exists!, do nothing.")    
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
#  retrieve BEC in h5py
import h5py
import numpy as np
path = 'c:\\Users\\Lab\\Desktop\\Data\\local\\0331\\BEC_LP10_121-121-121.h5'
with h5py.File(path, "r") as f:
    psiG = f['psiG'][()]
    psiE = f['psiE'][()]
    Nx = f['Metaparameters/Nx'][()]
    Ny = f['Metaparameters/Ny'][()]
    Nz = f['Metaparameters/Nz'][()]
    Lx = f['Metaparameters/Lx'][()]
    Ly = f['Metaparameters/Ly'][()]
    Lz = f['Metaparameters/Lz'][()]
    dw = f['Metaparameters/dw'][()]
    As = f['Parameters/As'][()]
    Nbec = f['Parameters/Nbec'][()]
    Rabi = f['Parameters/Rabi'][()]
    Wx = f['Parameters/Wx'][()]
    Wy = f['Parameters/Wy'][()]
    Wz = f['Parameters/Wz'][()]
    dw = f['Parameters/dw'][()]
    Ggg = f['Parameters/Ggg'][()]
    Gee = f['Parameters/Gee'][()]
    Gge = f['Parameters/Gge'][()]
    Geg = f['Parameters/Geg']
    print("retrieving succeeded!")
    xplot = np.linspace(-Lx,Lx,Nx)
    yplot = np.linspace(-Ly,Ly,Ny)
    zplot = np.linspace(-Lz,Lz,Nz)
    dxplot = np.diff(xplot)[0]
    dyplot = np.diff(yplot)[0]
    dzplot = np.diff(zplot)[0]
    Xplot,Yplot,Zplot = np.meshgrid(xplot,yplot,zplot)



#%%
print(np.sum(np.abs(TF_pbb)**2*dx*dy*dz))
print(np.sum(np.abs(psiG)**2*dxplot*dyplot*dzplot))
print(np.sum(np.abs(psiE)**2*dxplot*dyplot*dzplot))

#%%
import plotly.graph_objects as go
import matplotlib.pyplot as plt
plt.figure()
plt.plot(xplot, np.abs(psiG[61,:,61])**2*dxplot*dyplot*dzplot)
plt.xlabel('z')
plt.figure()
plt.plot(yplot, np.abs(psiG[:,61,61]**2)*dxplot*dyplot*dzplot)
plt.xlabel('y')
plt.figure()
plt.plot(zplot, np.abs(psiG[61,61,:]**2)*dxplot*dyplot*dzplot)
plt.xlabel('z')
plt.figure()
plt.plot(xplot, np.abs(psiE[61,:,61])**2*dxplot*dyplot*dzplot)
plt.xlabel('x')
plt.figure()
plt.plot(yplot, np.abs(psiE[:,61,61]**2)*dxplot*dyplot*dzplot)
plt.xlabel('y')
plt.figure()
plt.plot(zplot, np.abs(psiE[61,61,:]**2)*dxplot*dyplot*dzplot)
plt.xlabel('z')


#%%
print('\nisosurface\n')
import plotly.graph_objects as go
Data = np.abs(psiE**2*dxplot*dyplot*dzplot).flatten()
# [X,Y,Z] = np.meshgrid(x,y,z)
# Data = (X**2+Y**2+Z**2).flatten()
diff = np.max(Data) - np.min(Data)
fig= go.Figure(data=go.Isosurface(
    x=Xplot.flatten(),
    y=Yplot.flatten(),
    z=Zplot.flatten(),
    value = Data,
    # isomin=np.abs(Data[61,61,61]),
    # isomax=np.abs(Data[50,50,61]),
    isomin=np.min(Data) + diff/5,
    isomax=np.max(Data) - diff/5,
    # isomin=1e-5,
    # isomax=3e-5,
))
fig.show()

# import plotly.graph_objects as go

# fig= go.Figure(data=go.Isosurface(
#     x=[0,0,0,0,1,1,1,1],
#     y=[1,0,1,0,1,0,1,0],
#     z=[1,1,0,0,1,1,0,0],
#     value=[1,2,3,4,5,6,7,8],
#     isomin=2,
#     isomax=6,
# ))


# %% BEC intensity
cut = 10
import plotly.graph_objects as go
fig = go.Figure(data=[go.Surface(x=xplot,y=yplot,z=np.abs(psiG[:,:,cut]**2*dxplot*dyplot*dzplot))])
fig.update_layout(title='psiG_intensity', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()
fig = go.Figure(data=[go.Surface(x=xplot,y=yplot,z=np.abs(psiE[:,:,cut]**2*dxplot*dyplot*dzplot))])
fig.update_layout(title='psiE_intensity', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()

# %% BEC phase
cut = 10
import plotly.graph_objects as go
phase = np.arctan2(np.imag(psiG[:,:,cut])
                  ,np.real(psiG[:,:,cut]))
fig = go.Figure(data=[go.Surface(x=xplot,y=yplot,z=phase)])
fig.update_layout(title='psiG_phase', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()

phase = np.arctan2(np.imag(psiG[:,:,cut])
                  ,np.real(psiG[:,:,cut]))
fig = go.Figure(data=[go.Surface(x=xplot,y=yplot,z=phase)])
fig.update_layout(title='psiE_phase', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()
# %%

