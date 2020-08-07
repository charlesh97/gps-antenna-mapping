'''
========================
3D surface (solid color)
========================

Demonstrates a very basic plot of a 3D surface using a solid color.
'''
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as axes3d

x = np.linspace(0,2*np.pi, 400)
y = np.sin(x**2)

for i in range (1,10):
    ax1 = plt.subplot(4,4,i)
    ax1.plot(x,y)

plt.show()
print('finished')
theta, phi = np.linspace(0, .5*np.pi, 40), np.linspace(0, np.pi, 40)
THETA, PHI = np.meshgrid(theta, phi)
R = 1   #np.cos(PHI ** 2)
X = R * np.sin(PHI) * np.cos(THETA)
Y = R * np.sin(PHI) * np.sin(THETA)
Z = R * np.cos(PHI)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'), linewidth=0, antialiased=False, alpha=0.5)
plt.show()"""


import plotly.figure_factory as ff
import plotly.express as px

import numpy as np
#from scipy.spatial import Delaunay

"""u=np.linspace(-np.pi/2, np.pi/2, 60)
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
                         title="Boy's Surface")"""

#df = px.data.iris()
#fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
              #color='species')
#fig.show()



import plotly.graph_objects as go 
from plotly.subplots import make_subplots
print("starting")
x = np.linspace(0,10,2000)
y1 = np.sin(x)
y2 = np.sin(x)*np.cos(x)
fig = make_subplots(rows=2,cols=1)
fig.add_trace(
    go.Scatter(x=x,y=y1,mode='lines',name='sin'), row=1,col=1)
fig.add_trace(
    go.Scatter(x=x,y=y2,mode='lines',name='sincos'), row=1,col=1)
fig.add_trace(
    go.Scatter(x=x,y=y1,mode='lines',name='sin'), row=2,col=1)
fig.add_trace(
    go.Scatter(x=x,y=y2,mode='lines',name='sincos'), row=2,col=1)
print("finished")




