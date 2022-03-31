#%% compare two data, one from matlab, another from python.
base_name = 'C:\\Users\\Lab\\Desktop\\Data\\local\\test\\'
# matLG_name = r'testLG_03-29-2022_22-52.mat'
pyLG_name = r'LG10_121-121-121(8).h5'
# #%% retrieve local .mat files
# import scipy.io
# LGfile_matlab = scipy.io.loadmat(base_name + matLG_name)
# LGfile_matlab.keys()
# print(LGfile_matlab.get('L'))
# print(LGfile_matlab.get('P'))
# %%
# retrieve local h5py files
import h5py
import os
import numpy as np

base_dir = base_name
folder = ''
filename = pyLG_name
path = os.path.join(base_dir, folder, filename)

with h5py.File(path, "r") as f:
    LGdata_python = f['LGdata'][...]



# #%% data from matlab
# z = LGfile_matlab.get('Gridz')
# X = LGfile_matlab.get('X')
# Y = LGfile_matlab.get('Y')
# x = LGfile_matlab.get('Gridxy')
# y = LGfile_matlab.get('Gridxy')
# LGdata_matlab = LGfile_matlab['LGdata']
#%% LG intensity from python
cut = 1
import plotly.graph_objects as go
fig = go.Figure(data=[go.Surface(z=np.abs(LGdata_python[:,:,cut]))])
fig.update_layout(title=pyLG_name, autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()
# #%% LG intensity from matlab
# fig = go.Figure(data=[go.Surface(z=np.abs(LGdata_matlab[:,:,cut]))])
# fig.update_layout(title=matLG_name, autosize=False,
#                   width=500, height=500,
#                   margin=dict(l=65, r=50, b=65, t=90))
# fig.show()
# #%%
# np.isclose(LGdata_matlab,LGdata_python).all()


#%% LG phase from python
cut = 1
import plotly.graph_objects as go
phase = np.arctan2(np.imag(LGdata_python[:,:,cut])
                  ,np.real(LGdata_python[:,:,cut]))
fig = go.Figure(data=[go.Surface(z=phase)])
fig.update_layout(title=pyLG_name+' phase', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()
# #%% LG phase from matlab
# fig = go.Figure(data=[go.Surface(z=phase)])
# fig.update_layout(title=matLG_name+' phase', autosize=False,
#                   width=500, height=500,
#                   margin=dict(l=65, r=50, b=65, t=90))
# fig.show()
#%%






#%% plotting using matplotlib
# import numpy as np
# from matplotlib import pyplot as plt
# cut = 0 
# z = LGfile_matlab.get('Gridz')
# X = LGfile_matlab.get('X')
# Y = LGfile_matlab.get('Y')
# LGdata_matlab = LGfile_matlab['LGdata']
# plt.figure()
# plt.pcolor(X[:,:], Y[:,:], np.abs(LGdata_matlab[:,:,cut]))
# plt.figure()
# plt.pcolor(X[:,:], Y[:,:], np.abs(LGdata_python[:,:,cut]))