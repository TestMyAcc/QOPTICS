#%%
import numpy as np
import fdel2
Lx = 0.5
Ly = 0.5
Lz = 0.5
Nx = 4
Nz = 4
Ny = 4
x = np.linspace(-Lx,Lx,Nx)
y = np.linspace(-Ly,Ly,Ny)
z = np.linspace(-Lz,Lz,Nz)
dx = 2*Lx/(Nx-1)
dy = 2*Ly/(Ny-1)
dz = 2*Lz/(Nz-1)
[X,Y,Z] = np.meshgrid(x,y,z)
f = X**2 + Y**2 + Z**2
print(fdel2.del2_loop.__doc__)
print("==============")
print(fdel2.del2_ele.__doc__)



# %%
