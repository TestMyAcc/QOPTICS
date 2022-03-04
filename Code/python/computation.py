# %%
from numba import njit
import numpy as np

# %%
# Meta-parameters
nj = 10
stepJ = 1000   # counts of updating energy constraint
LG = 0
Nx = 252
Nz = 131
Ny = 252
Lx = 5
Ly = 4
Lz = 5
x = np.linspace(-Lx,Lx,Nx)
y = np.linspace(-Ly,Ly,Ny)
z = np.linspace(-Lz,Lz,Nz)
Nx,Ny,Nz = len(x), len(y), len(z)
[grid_x,grid_y,grid_z] = np.meshgrid(x,y,z)
pi = 3.14159265359
dx = np.diff(x)[0]
dy = np.diff(y)[0]
dz = np.diff(z)[0]
dw = 2e-6   # condition for converge : <1e-3*dx**2        

# Some constants
hbar = 2.054571800139113e-34 
m = 2.411000000000000e-25 # Rb atoms
unit = 1.222614572474304e-06 

# BEC parameters
As = 6.82e-09
Nbec = 10001
Rabi = 1001
Wx = 500
Wy = 500
Wz = 500
Rabi = Rabi/Wz                                                                               
Ggg = (4*pi*hbar**2*As*Nbec/m)*unit**-3*(hbar*Wz)**-1
Gee = Ggg  
Gge = 0
Geg = 0
Epot = ( (Wx**2*grid_x**2 + Wy**2*grid_y**2 + Wz**2*grid_z**2 )
            / (2*Wz**2) )
psiGmu = (15*Ggg / ( 16*pi*np.sqrt(2) )  )*(2/5)     # T.F. chemical energy (treat psiE0 as zeros)
psiEmu = (15*Gee / ( 16*pi*np.sqrt(2) )  )*(2/5)                                                  
TFsol = (psiGmu-Epot)/Ggg     #use                                
np.clip(TFsol, 0, np.inf,out=TFsol)
TFsol = np.sqrt(TFsol)

# # Laguerre-Gaussian light
# LGdata =  load(lightPath)
# Lambda = LGdata.Lambda
# L = LGdata.L
# P = LGdata.P
# W0 = LGdata.W0
# LGmsg = ['Using LG beam stored in:\n#s\nwith\n', 
#     'l,p=#d,#d Wavelength=#e BeamWasit=#e\n']
# fprintf(LGmsg,lightPath,L,P,Lambda,W0) #dimension needs care
# fprintf('\n')
# LG = 0.5*Rabi*LGdata.LGdata

# %%
# @njit(fastmath=True, nogil=True)
# def compute_BEC_Euler(j, nj):
#     """Calculating interaction between the BEC and L.G. beams.
#     Two-order system is used. The code evaluates ground-
#     state BEC and excited-state BEC, and save the data.
#     Note: Data is calculated without units. Use Euler method
#     to update time.
   
#     Args:
#         nj: Number of iterations.
#         stepJ: Number of iterations to update energy constraint.
#         isLight: interaction with light?.
#         x,y,z: coordinate vectors
#         dw: finite time difference.
#     """

    
#     for j in range(nj):
    
#         psiG = ( -dw*(  
#             -0.5 * laplacian(psiG,dx,dy,dz) +                      
#             ( Epot + Ggg*np.abs(psiG)**2 + Gge*np.abs(psiE)**2) * psiG  -       
#             psiGmu*psiG +                                                 
#             np.conjugate(LG)*psiE                                          
#             ) + psiG )
#         psiE = -dw*(  
#             -0.5 * laplacian(psiE,dx,dy,dz) +  
#             ( Epot  + Gee*np.abs(psiE)**2 + Geg*np.abs(psiG)**2)*psiE -  
#             psiEmu*psiE +  
#             LG*psiG  
#             ) + psiE

#         if (j % stepJ) == 0:
#         #  update energy constraint 
#             Nfactor = Normalize(psiG,psiE,dx,dy,dz)
#             J = J + 1 
#             psiGmu = psiGmu/(Nfactor)
#             psiEmu = psiEmu/(Nfactor)
#             psiGmuArray[J+1] = (np.sum(np.conjugate(psiG)*  
#                 (- 0.5 * 2*psiG*laplacian(psiG,dx,dy,dz) +     #Operated psiG
#                 (Epot + Ggg*np.abs(psiG)**2 + Gge*np.abs(psiE)**2) * psiG )  
#                 *dx*dy*dz))
#             psiEmuArray[J+1] = (np.sum(np.conjugate(psiE)*  
#                 (- 0.5 * 2*psiE*laplacian(psiE,dx,dy,dz) +     #Operated psiE
#                 ( Epot  + Gee*np.abs(psiE)**2 + Geg*np.abs(psiG)**2)*psiE) 
#                 *dx*dy*dz))


# %%
@njit(fastmath=True, nogil=True)
def compute_BEC_RK4():
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
    _psiEmu = psiEmu
    _psiE = np.zeros(TFsol.shape)
    _psiG = np.zeros(TFsol.shape)
    _psiG = TFsol
    
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
    tmp =  ( 0.5 * laplacian(_psiG,dx,dy,dz) 
         - (Epot + Ggg*np.abs(_psiG)**2 + Gge*np.abs(_psiE)**2)*_psiG
         + _psiGmu*_psiG )
    # set boundary points to zero
    tmp[0,:,:] = tmp[-1,:,:] = tmp[:,-1,:] = tmp[:,0,:] =  tmp[:,:,0] = tmp[:,:,-1] = 0
    return tmp
    
@njit(fastmath=True, nogil=True)
def E(_psiG, _psiE, _psiEmu):
    """Function of excited state"""
    tmp = np.zeros(_psiE.shape)
    tmp = ( 0.5 * laplacian(_psiE,dx,dy,dz) 
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
    psiG = _psiG + h/6 * ( k_g[...,0] + 2*k_g[...,1] + 2*k_g[...,2] + k_g[...,3] )
    psiE = _psiE + h/6 * ( k_e[...,0] + 2*k_e[...,1] + 2*k_e[...,2] + k_e[...,3] )

    return psiG, psiE
            
#%%
@njit(fastmath=True, nogil=True)
def laplacian(w, _dx, _dy, _dz):
    """ Calculate the laplacian of the array w=[].
        Note that the boundary points aren't handled"""
    laplacian = np.zeros(w.shape)
    for z in range(1,w.shape[2]-1):
        laplacian[:, :, z] = (1/_dz)**2 * ( w[:, :, z+1] - 2*w[:, :, z] + w[:, :, z-1] )
    for y in range(1,w.shape[0]-1):
        laplacian[y, ...] = laplacian[y, :, :] + (1/_dy)**2 * ( w[y+1, :, :] - 2*w[y, :, :] + w[y-1, :, :] )
    for x in range(1,w.shape[1]-1):
        laplacian[:, x,:] = laplacian[:, x, :] + (1/_dx)**2 * ( w[:, x+1, :] - 2*w[:, x, :] + w[:, x-1, :] )
    return laplacian


# %%
@njit(fastmath=True, nogil=True)
def Normalize(_psiG,_psiE,_dx,_dy,_dz):
    SumPsiG = np.sum( np.abs(_psiG)**2*_dx*_dy*_dz)
    SumPsiE = np.sum( np.abs(_psiE)**2*_dx*_dy*_dz)
    Nfactor = SumPsiG +  SumPsiE  
    return Nfactor



# %%
print("\n Total runs {} steps , update every {} steps\n".format(nj, stepJ))
psiG, psiE = compute_BEC_RK4()


# %%
