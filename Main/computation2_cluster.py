#%%
#BEGINNING(GPU calculation on cluster)
#Data are stored as 4-D array. The first dimension stores data of different parameter
import cupy as cp
import numpy as np
import h5py
import os
from scipy.special import genlaguerre


def computation(param,nj,stepJ,filename):
    base_dir = os.path.join(os.path.expanduser("~"),"Data")
    path = (os.path.join(base_dir, filename)) + '.h5'
    param = cp.array(param)
    # Meta-parameters and parameters
    Nx = 121
    Ny = 121
    Nz = 121
    Lx = 10
    Ly = 10
    Lz = 10
    x = cp.linspace(-Lx,Lx,Nx)
    y = cp.linspace(-Ly,Ly,Ny)
    z = cp.linspace(-Lz,Lz,Nz)
    dx = cp.diff(x)[0]
    dy = cp.diff(y)[0]
    dz = cp.diff(z)[0]
    dw = 1e-6   # condition for converge : <1e-3*dx**2

    [X,Y,Z] = cp.meshgrid(x,y,z)
    
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
    unit = cp.sqrt(hbar/m/Wz)

    Ggg = cp.array((4*pi*hbar**2*As*Nbec/m)*unit**-3*(hbar*Wz)**-1)
    Gee = cp.array(Ggg)
    Gge = cp.array(0)
    Geg = cp.array(0)
    Epot = ( (Wx**2*X**2 + Wy**2*Y**2 + Wz**2*Z**2 )
                / (2*Wz**2) )


    psiGmu = (15*Ggg / ( 16*pi*cp.sqrt(2) )  )**(2/5)  
    psiEmu = (15*Gee / ( 16*pi*cp.sqrt(2) )  )**(2/5) 

    # psiGmu = (15*Ggg/(64*cp.sqrt(2)*cp.pi))**(2/5) # for oval potential
    TF_amp = cp.array((psiGmu-Epot)/Ggg)
    cp.clip(TF_amp, 0, cp.inf,out=TF_amp)
    TF_pbb = cp.sqrt(TF_amp)
    total = cp.sum(cp.abs(TF_pbb)**2*dx*dy*dz)
    n_TF_pbb = TF_pbb/cp.sqrt(total,dtype=cp.complex128)

    psiG = cp.array(cp.abs(n_TF_pbb)*2,dtype=cp.complex128)
    psiE = cp.zeros_like(n_TF_pbb)
    # psiG = cp.array(cp.ones(TF_pbb.shape)+5,dtype=cp.complex128)
    # psiE = cp.array(cp.ones(TF_pbb.shape)+5,dtype=cp.complex128)

    # boradcast
    psiGmu = cp.repeat(psiGmu,cp.size(param))
    psiEmu = cp.repeat(psiEmu,cp.size(param))
    X = np.repeat(X[cp.newaxis,...],cp.size(param),axis=0)
    Y = np.repeat(Y[cp.newaxis,...],cp.size(param),axis=0)
    Z = np.repeat(Z[cp.newaxis,...],cp.size(param),axis=0)
    n_TF_pbb = np.repeat(n_TF_pbb[cp.newaxis,...],cp.size(param),axis=0)
    psiG = np.repeat(psiG[cp.newaxis,...],cp.size(param),axis=0)
    psiE = np.repeat(psiE[cp.newaxis,...],cp.size(param),axis=0)
    
    # Laguerre-Gaussian laser
    L = param
    W0 = 1
    Lambda = 1
    P = 0
    Zrl = cp.pi*W0**2/Lambda                         #Rayleigh length
    W= W0*cp.sqrt(1+(Z/Zrl)**2)  
    Rz = Z + Zrl**2/Z 
    Guoy = cp.einsum("i,ijkl->ijkl",(cp.abs(L)+2*P+1),cp.arctan2(Z,Zrl) )
    R = cp.sqrt(X**2 + Y**2)
    Phi = cp.arctan2(Y,X)
    AL = cp.array([data**(cp.abs(L[i])) for (i,data) in enumerate(cp.sqrt(2)*R/W)])
    ALpoly = cp.array([genlaguerre(P,np.abs(L.get()[i]))(2*(R.get()[i]/W.get()[i])**2) for i in range(cp.size(L)) ])
    AGauss = cp.exp(-(R/W)**2)
    Ptrans1 = cp.exp(-1j*(2*cp.pi/Lambda)*R**2/(2*Rz)) # Here
    Ptrans2 = cp.exp(-1j*  cp.einsum("i,ijkl->ijkl",L, Phi))
    PGuoy = cp.exp(1j*Guoy)
    LG = (W0/W)*AL*ALpoly*AGauss*Ptrans1*Ptrans2*PGuoy
    
    #TODO: correct the beam
    # if (L == 0 and P == 0):
    #     Plong = cp.exp(-1j*((2*cp.pi/Lambda)*Z - Guoy))
    #     LG = (W0/W)*AGauss*Ptrans1*Ptrans2*Plong
    LG = 1*LG/cp.max(cp.abs(LG)) 

    #TODO: return the updated energy
    # psiGmuArray = cp.zeros(int(nj/stepJ),dtype=cp.float32)
    # psiEmuArray = cp.zeros(int(nj/stepJ),dtype=cp.float32)
    # psiGmuArray[0] = psiGmu
    # psiEmuArray[0] = psiEmu
    J = 0

    for j in range(nj):
        Lap = cp.zeros_like(psiG) 
        
        if (j % stepJ) == 0 and j != 0:
            #  update energy constraint 
            SumPsiG = cp.sum( cp.abs(psiG)**2*dx*dy*dz, axis=(1,2,3))
            SumPsiE = cp.sum( cp.abs(psiE)**2*dx*dy*dz, axis=(1,2,3))
            Nfactor = SumPsiG +  SumPsiE  
            psiGmu = psiGmu/(Nfactor)
            psiEmu = psiEmu/(Nfactor)  
            J = J + 1
            # psiGmuArray[J] = psiGmu
            # psiEmuArray[J] = psiEmu
            with h5py.File(path, "w") as f:
                f['psiG'] = psiG.get()
                f['psiE'] = psiE.get()
                f['LG'] = LG.get()
                # f['psiGmuArray'] = psiGmuArray
                # f['psiEmuArray'] = psiEmuArray
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
                f['Parameters/Ggg'] = Ggg.get()
                f['Parameters/Gee'] = Gee.get()
                f['Parameters/Gge'] = Gge.get()
                f['Parameters/Geg'] = Geg.get()
                print("storing succeeded!")
            
            
        Lap[:, 1:Ny-1,1:Nx-1,1:Nz-1] = (
                (0.5/dy**2)*(
                        psiG[:, 2:Ny,   1:Nx-1, 1:Nz-1] 
                    - 2*psiG[:, 1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + psiG[:, 0:Ny-2, 1:Nx-1, 1:Nz-1])
                +(0.5/dx**2)*(
                        psiG[:, 1:Ny-1, 2:Nx,   1:Nz-1] 
                    - 2*psiG[:, 1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + psiG[:, 1:Ny-1, 0:Nx-2, 1:Nz-1])
                +(0.5/dz**2)*(
                        psiG[:, 1:Ny-1, 1:Nx-1, 2:Nz]
                    - 2*psiG[:, 1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + psiG[:, 1:Ny-1, 1:Nx-1, 0:Nz-2]))
        
        psiG_n = dw * (Lap - (Epot + Ggg*cp.abs(psiG)**2 + Gge*cp.abs(psiE)**2) * psiG \
                        - cp.conjugate(LG)*psiE + cp.einsum("i,ijkl->ijkl",psiGmu,psiG)) + psiG 
        

        Lap[:, 1:Ny-1,1:Nx-1,1:Nz-1] = (
                (0.5/dy**2)*(
                        psiE[:, 2:Ny,   1:Nx-1, 1:Nz-1] 
                    - 2*psiE[:, 1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + psiE[:, 0:Ny-2, 1:Nx-1, 1:Nz-1])
                +(0.5/dx**2)*(
                        psiE[:, 1:Ny-1, 2:Nx,   1:Nz-1] 
                    - 2*psiE[:, 1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + psiE[:, 1:Ny-1, 0:Nx-2, 1:Nz-1])
                +(0.5/dz**2)*(
                        psiE[:, 1:Ny-1, 1:Nx-1, 2:Nz]
                    - 2*psiE[:, 1:Ny-1, 1:Nx-1, 1:Nz-1] 
                    + psiE[:, 1:Ny-1, 1:Nx-1, 0:Nz-2]))
        psiE = dw * ( Lap - (Epot + Gee*cp.abs(psiE)**2 + Geg*cp.abs(psiG)**2) * psiE \
                        - LG*psiG  +cp.einsum("i,ijkl->ijkl",psiEmu,psiE)) + psiE
        psiG = psiG_n


if __name__ == "__main__":
    L = np.array([1])
    computation(L,10,10)