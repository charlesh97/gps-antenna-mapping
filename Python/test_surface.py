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

phi = np.linspace(0,np.pi,100)
theta = np.linspace(0, np.pi/2,100)
r = np.ones(100)

PHI,THETA = np.meshgrid(phi, theta)
x = r * np.sin(PHI) * np.cos(THETA)  
y = r * np.sin(PHI) * np.sin(THETA)
z = r * np.cos(PHI)
fig = go.Figure(
    go.Surface(x=x,y=y,z=z)
)
fig.show()
exit()



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

import numpy as np
a, b, d = 1.32, 1., 0.8
c = a**2 - b**2
u, v = np.mgrid[0:2*np.pi:100j, 0:2*np.pi:100j]
x = (d * (c - a * np.cos(u) * np.cos(v)) + b**2 * np.cos(u)) / (a - c * np.cos(u) * np.cos(v))
y = b * np.sin(u) * (a - d*np.cos(v)) / (a - c * np.cos(u) * np.cos(v))
z = b * np.sin(v) * (c*np.cos(u) - d) / (a - c * np.cos(u) * np.cos(v))

fig = make_subplots(rows=1, cols=2,
                    specs=[[{'is_3d': True}, {'is_3d': True}]],
                    subplot_titles=['Color corresponds to z', 'Color corresponds to distance to origin'],
                    )

fig.add_trace(go.Surface(x=x, y=y, z=z, colorbar_x=-0.07), 1, 1)
fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=x**2 + y**2 + z**2), 1, 2)
fig.update_layout(title_text="Ring cyclide")
fig.show()
exit()


#Grab data
theta, phi = np.linspace(0, .5*np.pi, 40), np.linspace(0, np.pi, 40)
THETA, PHI = np.meshgrid(theta, phi)
R = 1   #np.cos(PHI ** 2)
X = R * np.sin(PHI) * np.cos(THETA)
Y = R * np.sin(PHI) * np.sin(THETA)
Z = R * np.cos(PHI)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'), linewidth=0, antialiased=False, alpha=0.5)
plt.show()


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

