# %%
import numpy as np
import h5py
import dirutils,os

# %%
# retrieve local h5py files
base,filenames = dirutils.listLG() 
filename = input(f"Choose a filename from below:\n{filenames}")
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
    


#%% LG cross-line along xy-plane
from matplotlib import pyplot as plt

cut = 61
plt.figure()
plt.plot(x, np.abs(LGdata[61,:,cut])**2*dx*dy*dz)

#%% LG intensity from python
cut = 61
import plotly.graph_objects as go
fig = go.Figure(data=[go.Surface(z=np.abs(LGdata[:,:,cut]))])
fig.update_layout(title=filename, autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()

#%% LG phase from python
cut = 0
import plotly.graph_objects as go
phase = np.arctan2(np.imag(LGdata[:,:,cut])
                  ,np.real(LGdata[:,:,cut]))
fig = go.Figure(data=[go.Surface(z=phase)])
fig.update_layout(title=filename+' phase', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()






# %%
