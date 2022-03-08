#%%
import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import laguerre
from scipy.special import genlaguerre

def myLG(A,W0,Lambda,x,y,z,L,P):
#MYLG parallel loop version drawing the LG beam 
# spatial profile with determined waist and wavelength 
# along z-axis. Data is stored in specified path.
#
# MYLG(A,W0,Lambda,Gridz,Gridxy,L,P):
# Save data at displayed path.
# A: Amplitude of field.
# W0 : Beam Radius at z=0.
# Lambda: Wavelength of LG beam.
# Gridz : the coordinate point of the z point.
# Gridxy: 1-D array for x and y cooridnate.
# L: the azumithal mode number.
# P: the radial mode number.
# See also LaguerreGauss
# Algorithm faster than myLG.m in the factor of 178.~

#Beam parameters

    Zrl = np.pi*W0**2/Lambda                         #Rayleigh length
    W= W0*np.sqrt(1+(z/Zrl)**2)  
    # R = z + np.divide(Zrl**2, z, out=np.zeros_like(z), where=z!=0.0) #use numpy ufunc
    R = z + Zrl**2/z 
    Guoy = (abs(L)+2*P+1)*np.arctan2(z,Zrl) 
    
    Nx = np.size(x)
    Ny = np.size(y)
    Nz = np.size(z)
    [X,Y] = np.meshgrid(x, y)
    LGdata = np.zeros((Nx,Ny,Nz), dtype=np.cfloat)

    for k in range(Nz):
        LGdata[:,:,k] = computeLG(X,Y,z[k],W[k],R[k],Guoy[k],Lambda,L,P,W0)
    
    LGdata = A*LGdata/np.max(abs(LGdata)) 

    # tocBytes(gcp)
    # times = toc(tstart)
    time = 0
    # print("Beam waist W:{}, Wavelength lambda:{}".format(W0,Lambda))

    # datadir = '~/Documents/Lab/Projects/LGBeamdata/'
    # dlmt = '_'
    # fname = str(Nx)+dlmt+str(Ny)+dlmt+str(Nz)+'W0'+str(W0),dlmt,'Lambda',str(Lambda),dlmt,'L&P',str(L),str(P),'.mat'
    # save(strcat(datadir,fname),'LGdata','W0','Lambda','Gridxy','Gridz','L','P')
    
    return LGdata


def computeLG(X,Y,z,W,R,Guoy,Lambda,L,P,W0):
    r = np.squeeze(np.sqrt(X**2 + Y**2)) 
    Phi = np.arctan2(Y,X)
    AL =((np.sqrt(2)*r/W))**abs(L)
    ALpoly =genlaguerre(P,abs(L))(2*(r/W)**2)
    AGauss = np.exp(-(r/W)**2)
    Ptrans1 = np.exp(-1j*(2*np.pi/Lambda)*r**2/(2*R))
    Ptrans2 = np.exp(-1j*L*Phi)
    PGuoy = np.exp(1j*Guoy)
    LGdata = (W0/W)*AL*ALpoly*AGauss*Ptrans1*Ptrans2*PGuoy

    if (L == 0 and P == 0):
        Plong = np.exp(-1j*((2*np.pi/Lambda)*z - Guoy))
        LGdata = (W0/W)*AGauss*Ptrans1*Ptrans2*Plong
    return LGdata
#%%
def main():
    import numpy as np
    from matplotlib import pyplot as plt
    from numpy.polynomial import laguerre
    from scipy.special import genlaguerre
    #test numpy/scipy laguerre module
    # x = np.linspace(-5,15,1000)
    # y = np.linspace(-5,15,1000)
    # [X,Y] = np.meshgrid(x,y)
    # coef1 = np.array([1,1,0]).T
    # coef2 = np.array([1,0,1]).T
    # coef3 = np.array([0,1,1]).T
    # coefMtx = np.column_stack((coef1, coef2, coef3))
    # L = laguerre.lagval(X, coefMtx, tensor=Tr:ue)
    # # x = np.column_stack((X,X,X))
    # plt.plot(x,L.T)
    # plt.ylim([-10,20])
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
    # [grid_x,grid_y,grid_z] = np.meshgrid(x,y,z)
    dx = np.diff(x)[0]
    dy = np.diff(y)[0]
    dz = np.diff(z)[0]
    dw = 1e-6   # condition for converge : <1e-3*dx**2
    output = myLG(1,1,1,x,y,z,L=0,P=0)
    return output

#%%
if __name__ == "__main__":
    testLG = main()