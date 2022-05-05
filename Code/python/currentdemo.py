# test the current plot
#%%
import numpy as np
import matplotlib.pyplot as plt
import current as cur
import dirutils,os
import oam
import h5py

def curl(x,y,z,u,v,w):
    dx = x[0,:,0]
    dy = y[:,0,0]
    dz = z[0,0,:]

    dummy, dFx_dy, dFx_dz = np.gradient (u, dx, dy, dz, axis=[1,0,2])
    dFy_dx, dummy, dFy_dz = np.gradient (v, dx, dy, dz, axis=[1,0,2])
    dFz_dx, dFz_dy, dummy = np.gradient (w, dx, dy, dz, axis=[1,0,2])

    rot_x = dFz_dy - dFy_dz
    rot_y = dFx_dz - dFz_dx
    rot_z = dFy_dx - dFx_dy

    l = np.sqrt(np.power(u,2.0) + np.power(v,2.0) + np.power(w,2.0));

    m1 = np.multiply(rot_x,u)
    m2 = np.multiply(rot_y,v)
    m3 = np.multiply(rot_z,w)

    tmp1 = (m1 + m2 + m3)
    tmp2 = np.multiply(l,2.0)

    av = np.divide(tmp1, tmp2) # I don't know what av is.

    return rot_x, rot_y, rot_z, av

def A(X,Y,Z):
    """vector potential"""
    _Ax = np.zeros(X.shape)
    _Ay = (5/(np.sqrt(1+Y**2)))*np.arctan(X/(np.sqrt(1+Y**2)))
    _Az = np.zeros(Z.shape)
    return _Ax, _Ay, _Az


def test1():
    Nx = 301
    Ny = 301
    Nz = 201
    x = np.linspace(-1,1,Nx)
    y = np.linspace(-1,1,Ny)
    z = np.linspace(-1,1,Nz)
    X,Y,Z = np.meshgrid(x,y,z)
    Ax,Ay,Az = A(X,Y,Z)
    Bx,By,Bz,_ = curl(X,Y,Z,Ax,Ay,Az)

    cut = Nx//2
    # plt.quiver(x,y,Ax[:,:,cut],Ay[:,:,cut])
    numer = X**2 + Y**2 + 1
    Bz_ = 5/numer

    print(np.sum(np.abs(Bz - Bz_)/(Nx*Ny*Nz)))



base,filenames = dirutils.listLG() 
print(filenames)
filename = input(f"Choose a filename from below:\n{filenames}") + '.h5'
path = os.path.join(base, filename)

with h5py.File(path, "r") as f:
    LGdata = f['LGdata'][...]
    x = f['Coordinates/x'][()]
    y = f['Coordinates/y'][()]
    z = f['Coordinates/z'][()]
    W0 = f['Parameters/W0'][()]
    Lambda = f['Parameters/Lambda'][()]
    dx = np.diff(x)[0]
    dy = np.diff(y)[0]
    dz = np.diff(z)[0]


X,Y,Z = np.meshgrid(x,y,z)
dx = np.diff(x)[0]
dy = np.diff(y)[0]
dz = np.diff(z)[0]
# Jx, Jy, Jz = cur.current(LGdata, dx, dy, dz, m=1)
Lx, Ly, Lz = oam.oam(LGdata, dx, dy, dz, length=Lambda )
cut = 1
idx = np.arange(1,len(x),4)
idy = np.arange(1,len(y),4)
idz = np.arange(1,len(z),4)

plt.quiver(x,y,z,np.real(Lx[np.ix_(idx,idy)][...,cut]),
           np.real(Ly[np.ix_(idx,idy)][...,cut]),
           np.real(Lz[np.ix_(idx,idy)][...,cut]),
           angles='xy',
           units='x',
           scale_units='x',
           scale=0.09,
        #    headwidth=2,
        #    headlength=6,
           headaxislength=3,
           width=0.1
           )



# %%

# %%
