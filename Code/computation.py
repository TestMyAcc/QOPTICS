import sys
from scipy import ndimage 
import numpy as np
import mydel2


def compute_BEC(nj, stepJ, isLight):
    """Calculating interaction between BEC and L.G. beam.
    Two-order system is used. The code evaluates ground-
    state BEC and excited-state BEC.
   
    Args:
        nj: Number of iterations.
        stepJ: Number of iterations to update energy constraint.
        isLight: interaction with light?.
    """

    pi = 3.14159265359 
    Nx = 252 
    Nz = 131                                            
    Ny = 252 
    Lx = 5
    Ly = 4
    Lz = 5     # single side length                           
    x = np.linspace(-Lx,Lx,Nx)
    y = np.linspace(-Ly,Ly,Ny)
    z = np.linspace(-Lz,Lz,Nz)
    dx = 2*Lx/(Nx-1) 
    dy = 2*Ly/(Ny-1)
    dz = 2*Lz/(Nz-1)
    dw = 2e-6   # condition for converge : <1e-3*dx**2                                               

    # BEC parameters
    As = 6.82e-09
    Nbec = 10001
    Rabi = 1001
    hbar = 2.054571800139113e-34
    m = 2.411000000000000e-25
    omegaXY = 2001
    omegaZ = 2001
    unit = 1.222614572474304e-06

    # Dimensionlesslize 
    Rabi = Rabi/omegaZ                                                                               
    Ggg = (4*pi*hbar**2*As*Nbec/m)*unit**-3*(hbar*omegaZ)**-1
    Gee = Ggg  
    Gge = 0
    Geg = 0
    # LG and T.F approximation
    [grid_x,grid_y,grid_z] = np.meshgrid(x,y,z)

    psi_Gmu = (15*Ggg  /  (  16*pi*np.sqrt(2))  )*(2/5)     # T.F. chemical energy (treat psiE0 as zeros)
    psi_Emu = (15*Ggg  /  (  16*pi*np.sqrt(2))  )*(2/5)                                                  
    init_psiGmu = psi_Gmu
    init_psiEmu = psi_Emu

    Epot = 1/2 * (  (omegaXY/omegaZ)**2 *(grid_x*2 + grid_y*2) +  grid_z*2)
    TFsolG = (psi_Gmu-Epot)/Ggg     #use                                
    whereMinus = (TFsolG < 0)
    TFsolG[whereMinus] = 0  
    TFsolG = np.sqrt(TFsolG)

    if (isLight):
        # TODO: (L.G. beam): convert myLG.m to python code
        sys.exit()
        
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
    else:
        LG = 0

    psiGmuArray = np.zeros(1, np.ceil(nj/stepJ) + 1)   # Energy of groundstate every cetain step, 
    psiGmuArray[0] = init_psiGmu                 # At the first step is, the energy is the energy of T.F.          
    psiEmuArray = np.zeros(1, np.ceil(nj/stepJ) + 1)
    psiEmuArray[0] = init_psiEmu

    psiE = np.zeros([Nx,Ny,Nz]);  
    psiG = TFsolG;

    j = 0 
    J = 0                   
    print("\n Total runs {} steps , update every {} steps\n".format(nj, stepJ))

    while j != nj:

        j = j+1
        print("Current step: {}".format(j))

        # kinectic energy + potential and collision energy 
        # + constrain + light
        psiG = ( -dw*(  
            - 0.5 * ndimage.laplace(psiG) +                      
            ( Epot + Ggg*abs(psiG)**2 + Gge*abs(psiE)**2) * psiG  -          
            psi_Gmu*psiG +                                                    
            np.conjugate(LG)*psiE                                            
            ) + psiG )
        psiE = -dw*(  
            - 0.5 * ndimage.laplace(psiE) +  
            ( Epot  + Gee*abs(psiE)**2 + Geg*abs(psiG)**2)*psiE -  
            psi_Emu*psiE +  
            LG*psiG  
            ) + psiE

        if (j % stepJ) == 0:
        #  update energy constraint 
            Nfactor = Normalize(psiG,psiE,dx,dy,dz)
            J = J + 1 
            psi_Gmu = psi_Gmu/(Nfactor)
            psi_Emu = psi_Emu/(Nfactor)
            psiGmuArray[J+1] = (sum(np.conjugate(psiG)*  
                (- 0.5 * 2*psiG.ndim*ndimage.laplace(psiG) +     #Operated psiG
                (Epot + Ggg*abs(psiG)**2 + Gge*abs(psiE)**2) * psiG )  
                *dx*dy*dz,'all'))
            psiEmuArray[J+1] = (sum(np.conjugate(psiE)*  
                (- 0.5 * 2*psiE.ndim*ndimage.laplace(psiE) +     #Operated psiE
                ( Epot  + Gee*abs(psiE)**2 + Geg*abs(psiG)**2)*psiE) 
                *dx*dy*dz,'all'))


def Normalize(psiG,psiE,dx,dy,dz):
    SumPsiG = sum( abs(psiG)**2*dx*dy*dz,  'all' )
    SumPsiE = sum( abs(psiE)**2*dx*dy*dz,  'all' )
    Nfactor = SumPsiG +  SumPsiE  
    return Nfactor

    
if __name__ == '__main__':
    print("""\nUse "computeBEC()" as function!\n""")
    pass





