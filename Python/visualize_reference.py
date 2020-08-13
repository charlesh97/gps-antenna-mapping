"""
================================================================================================
This script will pull some reference data and display it using PyPlot
This is to be used as a test block and will be reimplemented in calculation.py
================================================================================================
"""
import os
import datetime
import pickle
import math
import csv
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import matplotlib.tri as mtri

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def main():
    #1300000000
    #1575000000
    file_in = "../Reference/ReferencePatchAntenna_v2.csv"
    Parse3DFile('1575000000', 'lin', file_in, "./ReferenceHorn2_Data.obj")
    data = pickle.load(open( "./ReferenceHorn2_Data.obj", "rb" ))
    theta = data.get('theta') * np.pi / 180
    phi = data.get('phi')  * np.pi / 180
    gain = data.get('gain') 

    offset = np.abs(np.min(gain))
    for i in range(len(gain)):
        for j in range(len(gain[i])):
            gain[i][j] = gain[i][j] + offset

    #print("Theta goes from", np.min(theta), "to", np.max(theta))
    #print("Phi goes from", np.min(phi), "to", np.max(phi))
    #print("Thetas: ", theta)

    #Convert to cartesian
    PHI, THETA = np.meshgrid(phi, theta)
    X = gain * np.sin(THETA) * np.cos(PHI)
    Y = gain * np.sin(THETA) * np.sin(PHI)
    Z = gain * np.cos(THETA)
    #X[60] = -X[60]

    # #Rotate around the y axis
    # alpha = -np.pi/2
    # x_ = X * np.cos(alpha) + Z * np.sin(alpha)
    # y_ = Y
    # z_ = X * -np.sin(alpha) + Z * np.cos(alpha)

    fig = go.Figure(
        go.Surface(x=X,y=Y,z=Z)
    )
    fig.update_layout(title=file_in)
    fig.show()


# Function: Parse3DFile()
# Details: Parses the 3d polar gain information from the near field chamber csv
# Parameters:
# frequency : int of frequency e.g. 1500000000
# gain_type : str of 'lin','rhcp','lhcp'
# file_in : csv file in
# file_out : obj out
def Parse3DFile(frequency, gain_type, file_in, file_out):
    phi_l = []
    theta_l = []
    gain_l = []

    col_idx = -1
    col_len = 121 #120 data points in the set

    if gain_type == 'lin':
        col_idx = 2
    if gain_type == 'rhcp':
        col_idx = 365
    if gain_type == 'lhcp':
        col_idx = 242
    if col_idx == -1:
        print('Problem with gain_type parameter. Please check parameter input.')
        return

    with open(file_in) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        start = False #This indicates if the script is actively pulling data (state machine)
        row_idx = 0
        line_count = 1
        for row in csv_reader:  #Loop through all rows
            if line_count == 2: #Grab all the theta_l values
                for j in range(col_len):
                    if j == 0:
                        theta_l.append(float(row[j+col_idx].split('=')[1]))
                    else:
                        theta_l.append(float(row[j+col_idx]))

            if not start and row[0] == frequency: #Find start of frequency (row)
                start = True

            if start and row[0] == frequency: #Running state - Right frequency
                phi_l.append(float(row[1]))
                gain_l.append([])

                for j in range(col_len):
                    gain_l[row_idx].append(float(row[j+col_idx]))

                row_idx = row_idx + 1

            if start and row[0] != frequency:
                #Need to append last bit data (copied from phi_l = 0)
                phi_l.append(np.pi)
                gain_l.append(np.flipud(gain_l[0][:]))
                break

            line_count = line_count + 1


    # Will convert to degrees now, easier to work with int
    for i in range(len(phi_l)):
        phi_l[i] = int(np.round(phi_l[i] / np.pi * 180))
    for j in range(len(theta_l)):
        theta_l[j] = int(np.round(theta_l[j] / np.pi * 180))


    # Shifting coordinate systems from NearField --> Spherical
    # Want to keep the data in a 2D array for later use
    # Start with changing PHI/THETA individually then rebuild 2D array
    data_points = []    #(theta, phi, gain) tuple
    theta_points = []   #Unorganized lists
    phi_points = []
    for i in range(len(gain_l)):
        for j in range(len(gain_l[i])):       
            theta = abs(theta_l[j])
            if theta_l[j] < 0:
                phi = phi_l[i] + 180
            else:
                phi = phi_l[i]
            
            if phi not in phi_points:     #Append to list of phi values
                phi_points.append(phi)
            if theta not in theta_points: #Append to list of theta values
                theta_points.append(theta)
            data_points.append((theta, phi, gain_l[i][j])) #Append gain datapoint

    #Rebuild array
    theta_points.sort()
    phi_points.sort()

    n = len(theta_points)
    m = len(phi_points)
    gain = [[np.nan for x in range(m)] for x in range(n)] #n x m matrix

    for k in data_points:
        n_i = theta_points.index(k[0])
        m_j = phi_points.index(k[1])
        gain[n_i][m_j] = k[2]

    for i in range(len(gain[0])):
        gain[0][i] = gain[0][0]

    data = {}
    data['theta'] = np.array(theta_points)
    data['phi'] = np.array(phi_points)
    data['gain'] = np.array(gain)
    pickle.dump(data, open(file_out, "wb"))
    print('3D Parsing Finished')




if __name__ == "__main__":
    main()