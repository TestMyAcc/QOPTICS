#%%
#  retrieve BEC in h5py
import h5py
import numpy as np
import dirutils,os

base_dir = os.path.join(os.path.expanduser("~"),"Data")
filenames = dirutils.lsfiles(base_dir) 
filename = input(f"Choose a filename from below:\n{filenames}") + '.h5'
path = os.path.join(base_dir, filename)
if not os.path.isfile(path):
    print("File isn't exist!")

#%%
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
    Geg = f['Parameters/Geg'][()]
    lgpath = f['LGfile'][()]
    print("retrieving succeeded!")

xplot = np.linspace(-Lx,Lx,Nx)
yplot = np.linspace(-Ly,Ly,Ny)
zplot = np.linspace(-Lz,Lz,Nz)
dxplot = np.diff(xplot)[0]
dyplot = np.diff(yplot)[0]
dzplot = np.diff(zplot)[0]
Xplot,Yplot,Zplot = np.meshgrid(xplot,yplot,zplot)
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
Epot = ( (Wx**2*Xplot**2 + Wy**2*Yplot**2 + Wz**2*Zplot**2 )
            / (2*Wz**2) )
# circular potential
psiGmu = (15*Ggg / ( 16*pi*np.sqrt(2) )  )**(2/5)  
psiEmu = (15*Gee / ( 16*pi*np.sqrt(2) )  )**(2/5) 

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
# psiGmu = (16*Ggg/(64*np.sqrt(2)*np.pi))**(2/5) # for oval potential
# %%
TF_amp = np.array((psiGmu-Epot)/Ggg,dtype=np.cfloat)
np.clip(TF_amp, 0, np.inf,out=TF_amp)
TF_pbb = np.sqrt(TF_amp)
total = np.sum(np.abs(TF_pbb)**2*dxplot*dyplot*dzplot)
n_TF_pbb = TF_pbb/np.sqrt(total)

#%%
print(np.sum(np.abs(TF_pbb)**2*dxplot*dyplot*dzplot))
print(np.sum(np.abs(psiG)**2*dxplot*dyplot*dzplot))
print(np.sum(np.abs(psiE)**2*dxplot*dyplot*dzplot))

#%%
import plotly.graph_objects as go
import matplotlib.pyplot as plt
print("abs|n_TF_pbb|^2")
print(np.sum(np.abs(n_TF_pbb)**2*dxplot*dyplot*dzplot))
print("abs|psiG|^2")
print(np.sum(np.abs(psiG)**2*dxplot*dyplot*dzplot))
print("abs|psiE|^2")
print(np.sum(np.abs(psiE)**2*dxplot*dyplot*dzplot))
#%%
plt.figure()
plt.plot(xplot, np.abs(n_TF_pbb[61,:,61])**2*dxplot*dyplot*dzplot)
plt.xlabel("xplot")
plt.title("n_TF_pbb")

plt.figure()
plt.plot(yplot, np.abs(n_TF_pbb[:,61,61]**2)*dxplot*dyplot*dzplot)
plt.xlabel("yplot")
plt.title("n_TF_pbb")

plt.figure()
plt.plot(zplot, np.abs(n_TF_pbb[61,61,:]**2)*dxplot*dyplot*dzplot)
plt.xlabel("zplot")
plt.title("n_TF_pbb")

plt.figure()
plt.plot(xplot, np.abs(psiG[61,:,61])**2*dxplot*dyplot*dzplot)
plt.xlabel("xplot")
plt.title("psiG")

plt.figure()
plt.plot(yplot, np.abs(psiG[:,61,61]**2)*dxplot*dyplot*dzplot)
plt.xlabel("yplot")
plt.title("psiG")

plt.figure()
plt.plot(zplot, np.abs(psiG[61,61,:]**2)*dxplot*dyplot*dzplot)
plt.xlabel("zplot")
plt.title("psiG")

plt.figure()
plt.plot(xplot, np.abs(psiE[61,:,61])**2*dxplot*dyplot*dzplot)
plt.xlabel("xplot")
plt.title("psiE")

plt.figure()
plt.plot(yplot, np.abs(psiE[:,61,61]**2)*dxplot*dyplot*dzplot)
plt.xlabel("yplot")
plt.title("psiE")

plt.figure()
plt.plot(zplot, np.abs(psiE[61,61,:]**2)*dxplot*dyplot*dzplot)
plt.xlabel("zplot")
plt.title("psiE")

plt.figure()
plt.plot(xplot, np.abs(LG[61,:,61])**2*dxplot*dyplot*dzplot)
plt.xlabel("xplot")
plt.title("LG")
#


#%%
print('\nisosurface\n')
# import plotly.graph_objects as go
# Data = np.abs(psiE**2*dxplot*dyplot*dzplot).flatten()
# # [X,Y,Z] = np.meshgrid(x,y,z)
# # Data = (X**2+Y**2+Z**2).flatten()
# diff = np.max(Data) - np.min(Data)
# fig= go.Figure(data=go.Isosurface(
#     x=Xplot.flatten(),
#     y=Yplot.flatten(),
#     z=Zplot.flatten(),
#     value = Data,
#     # isomin=np.abs(Data[61,61,61]),
#     # isomax=np.abs(Data[50,50,61]),
#     isomin=np.min(Data) + diff/5,
#     isomax=np.max(Data) - diff/5,
#     # isomin=1e-5,
#     # isomax=3e-5,
# ))
# fig.show()

# import plotly.graph_objects as go

# fig= go.Figure(data=go.Isosurface(
#     x=[0,0,0,0,1,1,1,1],
#     y=[1,0,1,0,1,0,1,0],
#     z=[1,1,0,0,1,1,0,0],
#     value=[1,2,3,4,5,6,7,8],
#     isomin=2,
#     isomax=6,
# ))


# %% intensity
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

fig = go.Figure(data=[go.Surface(x=xplot,y=yplot,z=np.abs(LG[:,:,cut]**2*dxplot*dyplot*dzplot))])
fig.update_layout(title='LG_intensity', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()
# %% phase
cut = 0
import plotly.graph_objects as go
phase = np.arctan2(np.imag(psiG[:,:,cut])
                  ,np.real(psiG[:,:,cut]))
fig = go.Figure(data=[go.Surface(x=xplot,y=yplot,z=phase)])
fig.update_layout(title='psiG_phase', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()

phase = np.arctan2(np.imag(psiE[:,:,cut])
                  ,np.real(psiE[:,:,cut]))
fig = go.Figure(data=[go.Surface(x=xplot,y=yplot,z=phase)])
fig.update_layout(title='psiE_phase', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()

phase = np.arctan2(np.imag(LG[:,:,cut])
                  ,np.real(LG[:,:,cut]))
fig = go.Figure(data=[go.Surface(x=xplot,y=yplot,z=phase)])
fig.update_layout(title='LG_phase', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()
