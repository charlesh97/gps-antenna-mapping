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
import matplotlib.pyplot as plt

theta, phi = np.linspace(0, 2*np.pi, 20), np.linspace(0, np.pi, 10)
THETA, PHI = np.meshgrid(theta, phi)
R = [[1 for x in range(20)] for x in range(10)] 
X = R * np.sin(PHI) * np.cos(THETA)
Y = R * np.sin(PHI) * np.sin(THETA)
Z = R * np.cos(PHI)

x = X.flatten()
y = Y.flatten()
z = Z.flatten()

#Prep for triangulation
points2D = []
for i in range(len(phi)):
    for j in range(len(theta)):
        points2D.append([phi[i],theta[j]])

#Now before you plot, delete some points
values = np.random.randint(0, 200, 100)
values.sort()
for val in reversed(values):
    points2D.pop(val)
    x = np.delete(x,val)
    y = np.delete(y, val)
    z = np.delete(z, val)

#Triangulation
points2D = np.array(points2D)
tri = Delaunay(points2D)
simplices = tri.simplices

fig = ff.create_trisurf(x=x, y=y, z=z,
                         colormap=['rgb(50, 0, 75)', 'rgb(200, 0, 200)', '#c8dcc8'],
                         show_colorbar=True,
                         simplices=simplices,
                         title="Boy's Surface")
fig.show()

"""

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plot = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'), linewidth=0, antialiased=False, alpha=0.5)
# plt.show()


#Delaunay
points = []
for i in range(len(phi)):
    for j in range(len(theta)):
        points.append([phi[i],theta[j]])
points = np.array(points)

tri = Delaunay(points)
plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')
plt.show()


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