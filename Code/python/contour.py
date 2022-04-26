# example code of isosurface plot.
#%%

import plotly.graph_objects as go

import pandas as pd

# Read data from a csv
z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')

fig = go.Figure(data=[go.Surface(z=z_data.values)])

fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

fig.show()

#%%
import plotly.graph_objects as go

fig= go.Figure(data=go.Isosurface(
    x=[0,0,0,0,1,1,1,1],
    y=[1,0,1,0,1,0,1,0],
    z=[1,1,0,0,1,1,0,0],
    value=[1,2,3,4],
    isomin=2,
    isomax=6,
))

fig.show()
# %%

import plotly.graph_objects as go
import numpy as np

X, Y, Z = np.mgrid[-5:5:40j, -5:5:40j, -5:5:40j]

# ellipsoid
values = X * X * 0.5 + Y * Y + Z * Z * 2

diff = np.max(values) - np.min(values)
fig = go.Figure(data=go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),    
    # isomin= (np.min(values) + diff/2.0),
    # isomax= (np.max(values) - diff/2.0),
    isomin= (np.max(values) - diff/2.0),
    isomax= (np.max(values) - diff/2.0),
    caps=dict(x_show=False, y_show=False,)
    ))
fig.show()
# %%
