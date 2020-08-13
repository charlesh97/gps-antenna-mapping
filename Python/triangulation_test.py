import os
import time as tm 
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


List_By_SV = pickle.load(open ( "list_by_sv.obj", "rb" ))
#### Plot combined 3D data - Lines
# Going to build an nxm array to store the list of R values for each SV <----- IMPORTANT DISTINCTION
# Average those values and build an unsorted list of phi/theta/r/time
# Sort the list by time
# Plot the lines with remaining phi/theta/r data
#maintitle = "Combined 3D SV Plot - " + file_in
#fig = go.Figure()

i = 0
theta_ = np.linspace(0,90,91)S
phi_ = np.linspace(0,359,360)
n = len(theta_)
m = len(phi_)
combined_r = [[ [] for x in range(m)] for x in range(n)] #n x m matrix

for key in List_By_SV:  
    # Now build the R nxm array
    r = [[[[],[]] for x in range(m)] for x in range(n)] #n x m matrix - [cno, time]
    for i in range(len(List_By_SV[key]["phi"])):
        x = List_By_SV[key]["theta"][i]
        y = List_By_SV[key]["phi"][i]
        r[x][y][0].append(List_By_SV[key]["cno"][i])
        r[x][y][1].append(List_By_SV[key]["time_idx"][i])
        combined_r[x][y].append(List_By_SV[key]["cno"][i])   #This is used in the next portion

    # Average Values
    # data = {"phi":[],"theta":[],"r":[],"time":[]} #phi,theta,r,time_idx
    # for i in range(n):
    #     for j in range(m):
    #         if len(r[i][j][0]) > 0:
    #             data["theta"].append(i * np.pi / 180) #i is the same as phi(i)
    #             data["phi"].append(j * np.pi / 180)
    #             data["r"].append(np.mean(r[i][j][0]))
    #             data["time"].append(int(np.mean(r[i][j][1])))
                
    # # Use dataframe pandas to sort by time
    # df = pd.DataFrame(data)
    # df = df.sort_values(by='time').reset_index(drop=True)   #don't forget to reindex
    # x = df["r"] * np.sin(df["theta"]) * np.cos(df["phi"])      #sin(PHI), cos(THETA)       #WHERE THETA @ Z-AXIS = 0, @XY-PLANE = 90
    # y = df["r"] * np.sin(df["theta"]) * np.sin(df["phi"])
    # z = df["r"] * np.cos(df["theta"])
    # fig.add_trace(
    #     go.Scatter3d(x=x, y=y, z=z, mode='markers', name=str(key))
    # )
#f = "../Exported Plots/" + file_in.split('.')[0] + "-3DCombinedSV.png"
#fig.update_layout(title_text=maintitle)
#fig.update_layout(height=1440,width=2560)
#fig.show()


#### Plot combined 3D data - Surface
# combined_r already has nxm (phi/theta) averaged data
# Build x,y,z arrays
#
#maintitle = "Surface 3D SV Plot - " + file_in
data = {"phi":[],"theta":[],"r":[],"time":[]}

theta = np.linspace(0,90,91) * np.pi / 180
phi = np.linspace(0,359,360) * np.pi / 180
R = [[ np.nan for x in range(m)] for x in range(n)] #n x m matrix

# Average all the values, find the minimum
# rmin = 1000
for i in range(n):
    for j in range(m):
        if len(combined_r[i][j]) > 0: #empty list otherwise
            R[i][j] = (np.mean(combined_r[i][j]))
            
            # if R[i][j] < rmin:
            #     rmin = R[i][j]

# SCRATCH THIS
# NO OFFSET BECAUSE CNO IS ALWAYS POSITIVE FOR GPS
# Add the offset
# offset = np.abs(rmin)
# for i in range(n):
#     for j in range(m):
#         if not np.isnan(R[i][j]): #Not empty list
#             R[i][j] = R[i][j] + offset

PHI, THETA = np.meshgrid(phi, theta)
X = R * np.sin(THETA) * np.cos(PHI)
Y = R * np.sin(THETA) * np.sin(PHI)
Z = R * np.cos(THETA)

#Build the lists for triangulation
#Get rid of the np.nan / missing values in 2D array
x = []
y = []
z = []
points2D = []
r = []
for i in range(n):
    for j in range(m):
        if not np.isnan(R[i][j]):
            points2D.append([theta[i], phi[j]])
            x.append(X[i][j])
            y.append(Y[i][j])
            z.append(Z[i][j])
            r.append(R[i][j])

points2D = np.array(points2D)
tri = Delaunay(points2D)
plt.triplot(points2D[:,0],points2D[:,1], tri.simplices)
plt.plot(points2D[:,0],points2D[:,1],'o')
#plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x,y,z)
#plt.show()

# Go through the simplices and map the triangle point indexes to the R values and average
# Simplice points are indexes to the Point2D list. Point2D are (theta,phi) values that are indexes to the nxm R array
simp = tri.simplices
color_map = []
for tri in simp:
    idx_list = points2D[tri][:]                                     #Will generate a list of three points in points2D (indexes)
    idx_list = np.round((idx_list / np.pi * 180)).astype('int')     #Converting radians to degrees. The degrees map to the indexes of theta/phi
    
    tmp_r = []
    for idx in idx_list:                                            #Map the index from the idx_list (point2D array conveted)
        t = idx[0]
        p = idx[1]
        tmp_r.append(R[t][p])
    
    tmp_r = np.min(tmp_r)                                  #Average the three values from the triangle and append to the color map for that point.
    color_map.append(tmp_r)

print("Ready to plot")
fig = ff.create_trisurf(x=x,y=y,z=z,simplices=simp,title="Triangular Mesh", colormap='Rainbow', color_func=color_map)
fig.show()