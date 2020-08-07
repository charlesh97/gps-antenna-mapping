import os
import datetime
import pickle
import math
import csv
import numpy as np
import pandas as pd

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.spatial import Delaunay
"""
u=np.linspace(-np.pi/2, np.pi/2, 60)
v=np.linspace(0, np.pi, 60)
u,v=np.meshgrid(u,v)
u=u.flatten()
v=v.flatten()

x = (np.sqrt(2)*(np.cos(v)*np.cos(v))*np.cos(2*u) + np.cos(u)*np.sin(2*v))/(2 - np.sqrt(2)*np.sin(3*u)*np.sin(2*v))
y = (np.sqrt(2)*(np.cos(v)*np.cos(v))*np.sin(2*u) - np.sin(u)*np.sin(2*v))/(2 - np.sqrt(2)*np.sin(3*u)*np.sin(2*v))
z = (3*(np.cos(v)*np.cos(v)))/(2 - np.sqrt(2)*np.sin(3*u)*np.sin(2*v))

points2D = np.vstack([u, v]).T
tri = Delaunay(points2D)
simplices = tri.simplices

fig = ff.create_trisurf(x=x, y=y, z=z,
                         colormap=['rgb(50, 0, 75)', 'rgb(200, 0, 200)', '#c8dcc8'],
                         show_colorbar=True,
                         simplices=simplices,
                         title="Boy's Surface")
fig.show()"""




#Grab data
data = pickle.load(open ( "dataset.obj", "rb" ))
df = pd.DataFrame(data)
df = df.sort_values(by=['phi','theta']).reset_index(drop=True)
x = df["r"] * np.sin(df["phi"]) * np.cos(df["theta"])      #sin(PHI), cos(THETA)       #WHERE PHI @ Z-AXIS = 0, @XY-PLANE = 90
y = df["r"] * np.sin(df["phi"]) * np.sin(df["theta"])
z = df["r"] * np.cos(df["phi"])
fig = go.Figure(
    go.Mesh3d(x=x,y=y,z=z,alphahull=15,opacity=0.7)
    #go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=8,color=df["r"], colorbar=dict(title="Colorbar"), colorscale='Inferno',opacity=0.8))
)
fig.show()
"""#Dataframe sorting
df = pd.DataFrame(data)
df = df.sort_values(by=['phi','theta']).reset_index(drop=True)

PHI,THETA = np.meshgrid(df["phi"], df["theta"])
PHI = PHI.flatten()
THETA = THETA.flatten()
R = (df["r"]**df["r"])
x = df["r"] * np.sin(PHI) * np.cos(THETA)  
y = df["r"] * np.sin(PHI) * np.sin(THETA)
z = df["r"] * np.cos(PHI)

points2D = np.vstack([PHI,THETA]).T
tri = Delaunay(points2D)
simplices = tri.simplices

fig = ff.create_trisurf(x=x,y=y,z=z,simplices=simplices)
fig.show()"""

