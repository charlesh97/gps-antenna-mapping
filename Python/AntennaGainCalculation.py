"""
================================================================================================
This script will pull the raw GPS data and calculate the antenna gain of an unknown antenna
based on 3D gain data of the reference antenna.

Both antennas are taking data simultaneously. Data is aligned by satellite and timestamp
================================================================================================
"""
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
from micropyGPS import MicropyGPS

import plotly.figure_factory as ff
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy import signal

from tqdm import tqdm

def main():
    starttime = tm.time()

    #### Parse raw data ####
    GPS_Parse_SV("../Logged/Dipole-8-5-20.TXT", "Dipole_Raw_GPS.obj")
    GPS_Parse_SV("../Logged/Patch-8-5-20.TXT", "Ref_Raw_GPS.obj")
    #Parse3DFile('1580000000', 'rhcp', "../Reference/ReferencePatchAntenna.csv", "./ReferencePatch_Data.obj")
    #PlotReference("ReferencePatch_Data.obj", True)

    #### Time align data, remove bad SVs ####
    MatchUpData("Dipole_Raw_GPS.obj", "Ref_Raw_GPS.obj")

    #### Clean up data  ####
    CleanData("Dipole_Raw_GPS.obj", "Dipole_Time.obj") 
    CleanData("Ref_Raw_GPS.obj", "Ref_Time.obj")

    #### Combine the data and prep for delta calculations ####
    CalculateDelta("Dipole_Time.obj", "Ref_Time.obj", "DeltaData.obj")

    #### Plot and save data ####
    #PlotDeltaCNO("Dipole_SV.obj", "Ref_SV.obj", True)
    Plot_Surface("DeltaData.obj")
    Plot_Scatter("DeltaData.obj")
    #### Combine the data and prep for delta calculations ####
    #CombineData("Dipole_Time.obj", "Ref_Time.obj", "CompiledData.obj")
    #Calculate Antenna Gain()
    
    #Print("CompiledData.obj",  "ReferenceTurnstile_Data.obj", "FinalAntennaPattern.obj")

    seconds = tm.time() - starttime
    print("main finished in " + str(seconds) + " seconds")

##############################################################################################################
# ANTENNA CALCULATION FUNCTIONS
##############################################################################################################
#Function: MatchUpData()
#Description: Time aligns data
def MatchUpData(file1_in, file2_in):
    data1 = pickle.load(open( file1_in, "rb" )) #Dipole
    data2 = pickle.load(open( file2_in, "rb" )) #Turnstile

    #Convert to numpy array first
    data1 = np.array(data1)
    data2 = np.array(data2)

    ####### BEFORE DATA ALIGN #########
    # Remove any missing datapoints
    # Adjust the GPS elevation and azimuth to match spherical coordinates phi and theta
    ###################################
    # Spherical Coordinates Convention Used
    #         THETA=0
    #           │      / 
    #           │    /
    #           │  /
    #           │/
    # ----------│------------ THETA=90, PHI=90
    #         / │
    #       /   │
    #     /     │
    #   THETA=90
    #   PHI=0
    ###################################

    ##########################################
    #Find timestamp differences - Array #1 
    ##########################################
    totalRemoved1 = 0
    totalRemoved2 = 0

    remove1 = []
    remove2 = []
    timestamp_list = data2[:,0]
    for idx in range(len(data1)):               #Search by timestamp from the top
        if data1[idx][0] not in timestamp_list:
            remove1.append(idx)
        else:
            break
    for idx in reversed(remove1):               # Remove missing timestamps              
        data1 = np.delete(data1, idx, 0)
    
    remove1.clear() #Don't forget to clear
    for idx in reversed(range(len(data1))):     #Search by timestamp from the bottom
        if data1[idx][0] not in timestamp_list:
            remove1.append(idx)
        else:
            break
    for idx in remove1:                         #Remove missing timestamps (not reversed, you searched from the bottom)
        data1 = np.delete(data1, idx, 0)

    ##########################################
    #Find timestamp differences - Array #2 
    ##########################################
    timestamp_list = data1[:,0]
    for idx in range(len(data2)):               #Search by timestamp from the top
        if data2[idx][0] not in timestamp_list:
            remove2.append(idx)
        else:
            break
    for idx in reversed(remove2):               # Remove missing timestamps              
        data2 = np.delete(data2, idx, 0)

    remove2.clear() #Don't forget to clear
    for idx in reversed(range(len(data2))):     #Search by timestamp from the bottom
        if data2[idx][0] not in timestamp_list:
            remove2.append(idx)
        else:
            break
    for idx in remove2:                         #Remove missing timestamps (not reversed, you searched from the bottom)
        data2 = np.delete(data2, idx, 0)

    ##########################################
    #Find timestamp differences - Manually
    ##########################################
    remove1.clear()
    remove2.clear()

    if len(data1) != len(data2):                            #Only do this if the lengths are already equal
        
        print("Found extra timestamps in dataset...")
        while(len(data1) != len(data2)):                    #There may be multiple sections of missing data so continue through the whole array

            for idx in range(min(len(data1),len(data2))):   #Get where it starts
                if data1[idx][0] != data2[idx][0]:
                    mismatch_idx = idx
                    print("Found mismatch at",mismatch_idx,data1[mismatch_idx][0],data2[mismatch_idx][0])
                    break
            
            if len(data1) > len(data2):                     #Delete from the larger array   
                for idx in range(mismatch_idx, len(data1)):
                    if data1[idx][0] != data2[mismatch_idx][0]:
                        remove1.append(idx)
                        print("Appended index",idx,"to remove1")
                    else:
                        print("Found a match to remove1 at", idx)
                        break
            else:
                for idx in range(mismatch_idx, len(data2)):
                    if data2[idx][0] != data1[mismatch_idx][0]:
                        remove2.append(idx)
                        print("Appended index",idx,"to remove2")
                    else:
                        print("Found a match to remove2 at", idx)
                        break
            
            if len(remove1) > 0:                            #Delete the indexes you found
                for idx in reversed(remove1):
                    data1 = np.delete(data1, idx, 0)
            if len(remove2) > 0:
                for idx in reversed(remove2):
                    data2 = np.delete(data2, idx, 0)

        print("Extra timestamps cleaned.")

    print("Removed " + str(len(remove1)) + " " + str(len(remove2)) + " timestamp datapoints")
    totalRemoved1 = totalRemoved1 + len(remove1)
    totalRemoved2 = totalRemoved2 + len(remove2)

    ##########################################
    #Find missing datapoints - Array #1
    ##########################################
    remove1.clear()
    remove2.clear()
    for idx in range(len(data1)):               #By timestamp
        for key in data1[idx][1]:               #By satellite ID
            data1[idx][1][key] = list(data1[idx][1][key])   #<----- Sneak a tuple to list conversion here for later
            val = data1[idx][1][key]
            if val[0] == None or val[1] == None or val[2] == None:  #Remove bad data
                remove1.append([idx, key])
                if key in data2[idx][1]:                            #Remove data from other side as well if exists
                    remove2.append([idx, key])                     
    
    for x in reversed(remove1):
        data1[x[0]][1].pop(x[1])
    for x in reversed(remove2):
        data2[x[0]][1].pop(x[1])

    ##########################################
    #Find missing datapoints - Array #2
    ##########################################
    remove2.clear()
    for idx in range(len(data2)):               #By timestamp
        for key in data2[idx][1]:               #By satellite ID
            data2[idx][1][key] = list(data2[idx][1][key])   #<----- Sneak a tuple to list conversion here for later
            val = data2[idx][1][key]
            if val[0] == None or val[1] == None or val[2] == None:
                remove2.append([idx, key])      #Already searched first array for Nones
    for x in reversed(remove2):
        data2[x[0]][1].pop(x[1])
    
    print("Removed " + str(len(remove1)) + " " + str(len(remove2)) + " empty satellite datapoints")
    totalRemoved1 = totalRemoved1 + len(remove1)
    totalRemoved2 = totalRemoved2 + len(remove2)

    ###############################################
    #Find missing satellite datapoints - Array #1
    ###############################################
    remove1.clear()
    remove2.clear()
    for idx in range(len(data1)):
        for key in data1[idx][1]:
            if key not in data2[idx][1]:
                remove1.append([idx, key])
    for x in reversed(remove1):
        data1[x[0]][1].pop(x[1])
    
    for idx in range(len(data2)):
        for key in data2[idx][1]:
            if key not in data1[idx][1]:
                remove2.append([idx, key])
    for x in reversed(remove2):
        data2[x[0]][1].pop(x[1])

    print("Removed " + str(len(remove1)) + " " + str(len(remove2)) + " missing satellite datapoints")
    totalRemoved1 = totalRemoved1 + len(remove1)
    totalRemoved2 = totalRemoved2 + len(remove2)

    #Check that the length of timestamps are the same
    if len(data1) != len(data2):
        print("Data1:" + str(len(data1)) + " Data2:" + str(len(data2)) + " mismatch!")
        exit()
    
    #Check that the number of satellite datapoints are the same
    total_count = 0
    length1 = {}
    for idx in range(len(data1)):
        for key in data1[idx][1]:
            total_count = total_count + 1
            if key not in length1:
                length1[key] = 0
            else:
                length1[key] = length1[key] + 1
    length2 = {}
    for idx in range(len(data2)):
        for key in data2[idx][1]:
            if key not in length2:
                length2[key] = 0
            else:
                length2[key] = length2[key] + 1

    for l in length1:
        if length1[l] != length2[l]:
            print("-------ERROR------")
            print("Misaligned datasets found at SV:" + str(l) + ", " + str(length1[l]) + ", " + str(length2[l]))
            exit()
        #print("SV:" + str(l) + "\t-\t" + str(length1[l]) + ", " + str(length2[l]))
    
    print(str(total_count) + " individual datapoints, " + str(len(length1)) + " SVs")

    pickle.dump(data1, open( file1_in, "wb"))
    pickle.dump(data2, open( file2_in, "wb"))
    print("Finished time align")

#Function: CleanData()
#Description: Will clean data up, handle rotations. File_out is time organized
def CleanData(file_in, file_out):
    #In order to clean the data, we need to seperate the SVs (keep timestamps), and then smooth the phi, cno, az
    data = pickle.load(open ( file_in, "rb" ))

    #First adjust the elev/azimuth values to match spherical coordinate system
    for idx in range(len(data)):               #By timestamp
        for key in data[idx][1]:               #By satellite ID
            val = data[idx][1][key]
            val[0] = 90 - val[0] #Convert to elev to THETA
            if val[0] > 90:
                val[0] = 90

            val[1] = (360 - val[1]) % 360 #Azimuth to PHI

    #Reorganizing the list by SV, not time
    List_By_SV = {}
    with tqdm(total=len(data), desc="Reorganizing by SV [" + file_in + "]") as t:
        for idx in range(len(data)):               #By timestamp
            for key in data[idx][1]:               #By satellite ID
                val = data[idx][1][key][:]
            
                if key not in List_By_SV:          
                    List_By_SV[key] = {}
                if "phi" not in List_By_SV[key]:
                    List_By_SV[key]["phi"] = []
                if "theta" not in List_By_SV[key]:
                    List_By_SV[key]["theta"] = []
                if "cno" not in List_By_SV[key]:
                    List_By_SV[key]["cno"] = []
                if "time" not in List_By_SV[key]:
                    List_By_SV[key]["time"] = []
                if "time_idx" not in List_By_SV[key]:
                    List_By_SV[key]["time_idx"] = []

                List_By_SV[key]["theta"].append(val[0])
                List_By_SV[key]["phi"].append(val[1])
                List_By_SV[key]["cno"].append(val[2])
                List_By_SV[key]["time"].append(data[idx][0])    
                List_By_SV[key]["time_idx"].append(idx)   #Add timestamp(index) for future reference
            t.update()

    #Each key now has lists -> convert to arrays
    for key in List_By_SV:
        List_By_SV[key]["phi"] = np.array(List_By_SV[key]["phi"])   #Convert to array
        List_By_SV[key]["theta"] = np.array(List_By_SV[key]["theta"])
        List_By_SV[key]["cno"] = np.array(List_By_SV[key]["cno"])


    #Continue cleaning the data
    #Smooth and save data 
    # PEFORM A LOWPASS FILTER
    # b_filter, a_filter = signal.butter(2, 0.0005, 'lowpass')

    # for satnum in List_By_SV:
    #     filter_input = pd.Series(List_By_SV[satnum]["cno"])
    #     if(len(filter_input) > 500):
    #         padlength = 500
    #     else:
    #         padlength = len(filter_input)-1
    #     filter_output = signal.filtfilt(b_filter, a_filter, filter_input, padlen=padlength, padtype ='even')
    #     List_By_SV[satnum]["cno"] = filter_output

    # Now save the data back - both original timestamp and new satellite ID keyed
    # ['Timestamp', {SV Information}] -- Used later to reorganize into dictionary
    data_to_save = [[None, {}] for x in range(len(data))]
    with tqdm(total=len(List_By_SV), desc="Saving Modified [" + file_in + "]") as t:
        for key in List_By_SV:
            for time_idx in List_By_SV[key]["time_idx"]:
                if key not in data_to_save[time_idx][1]:
                    data_to_save[time_idx][1][key] = []

                idx = List_By_SV[key]["time_idx"].index(time_idx)
                data_to_save[time_idx][1][key] = [List_By_SV[key]["theta"][idx], List_By_SV[key]["phi"][idx], List_By_SV[key]["cno"][idx]]

                if data_to_save[time_idx][0] == None:
                    data_to_save[time_idx][0] = List_By_SV[key]["time"][idx]
            t.update() #track progress
        
    pickle.dump(data_to_save, open (file_out, "wb"))
    print("Finished cleaning " + file_in)

#Function: CalculateDelta()
#Description:   Combines both datasets (pre-lined up in MatchUpData), and calculates the delta SNR. 
#               Exports an object with the phi, theta, and 2D-gain array
def CalculateDelta(file1_in, file2_in, file_out):
    data1 = pickle.load(open( file1_in, "rb" )) #Test Antenna
    data2 = pickle.load(open( file2_in, "rb" )) #Reference

    # First build the 1xn & 1xm arrays for theta and phi, respectively
    # This is fuckin dumb, just map the whole range
    theta = np.linspace(0,90,91) #TODO: CHECK IF ALL ARRAYS ARE 91 LONG
    phi = np.linspace(0,360,361)

    # Now build the R nxm array.
    n = len(theta)
    m = len(phi)
    r = [[None for x in range(m)] for x in range(n)] #n x m matrix

    total = 0
    unique = 0

    # Put the data into the nxm array 
    with tqdm(total=len(data1), desc="Building 2D Gain Array") as bar:
        for time_idx in range(len(data1)):       #List of timestamps
            for key in data1[time_idx][1]:                    #List of SVs
                t = data1[time_idx][1][key][0]                #Remember data1: [[timestamp, {SV information}],...]
                p = data1[time_idx][1][key][1]
                d = data1[time_idx][1][key][2] - data2[time_idx][1][key][2]

                # Think about calculating the intermediate elev/az here
                n_i = t#theta.index(t)
                m_j = p#phi.index(p)

                if r[n_i][m_j] == None:
                    r[n_i][m_j] = []
                
                r[n_i][m_j].append(d)
                total = total + 1
            bar.update()

    # Average the delta values now
    for i in range(len(r)):
        for j in range(len(r[i])):
            if r[i][j] != None:
                r[i][j] = np.mean(r[i][j])
                unique = unique + 1
            else:
                r[i][j] = np.nan
    print(unique,"unique points found of",total)
    # Store data
    data = {}
    data['r'] = r
    data['phi'] = np.array(phi)
    data['theta'] = np.array(theta)
    pickle.dump(data, open(file_out, "wb"))
    print("Saved",file_out)

#Function: Calculate3D()
#Description: Will calculate the final antenna gain and plot.
def Calculate_3D(file_in, reference_in, file_out):
    # Grab data from serialized object
    data = pickle.load(open( file_in, "rb" ))
    ref = pickle.load(open( reference_in, "rb" ))

    data_theta = data.get('theta')
    data_phi = data.get('phi')
    data_delta = data.get('r')

    ref_theta = ref.get('theta')
    ref_phi = ref.get('phi')
    ref_gain = ref.get('gain')

    # Calculate the gain of the device under test
    # Generate a new array
    plot_gain = data_delta[:][:]

    # For each data_delta value, find the index and corresponding phi/theta
    for i in range(len(data_phi)):
        for j in range(len(data_theta)):
            if data_delta[i][j] != None:
                #Search phi
                phi_idx1 = 0
                phi_idx2 = 0 
                if data_phi[i] not in ref_phi:
                    #Search for the closest
                    for k in range(len(ref_phi)):
                        if data_phi[i] > ref_phi[k]:
                            phi_idx1 = k        #If idx1 is the last element in the array, idx2 will automatically be element 0 (It needs to be)
                        else:
                            phi_idx2 = k
                            break
                else:
                    phi_idx1 = ref_phi.index(data_phi[i])
                    phi_idx2 = phi_idx1

                #Search theta
                thta_idx1 = 0
                thta_idx2 = 0 
                if data_theta[j] not in ref_theta:
                    #Search for the closest
                    for k in range(len(ref_theta)):
                        if data_theta[j] > ref_theta[k]:
                            thta_idx1 = k
                        else:
                            thta_idx2 = k
                            break
                else:
                    thta_idx1 = ref_theta.index(data_theta[j])
                    thta_idx2 = thta_idx1
                
                #Interpolated in y
                if ref_phi[phi_idx2]-ref_phi[phi_idx1] == 0:
                    g1 = ref_gain[phi_idx1][thta_idx1]
                    g2 = ref_gain[phi_idx1][thta_idx2]
                else:
                    dy1 = ref_gain[phi_idx2][thta_idx1]-ref_gain[phi_idx1][thta_idx1]
                    dy2 = ref_gain[phi_idx2][thta_idx2]-ref_gain[phi_idx1][thta_idx2]
                    g1 = (i-ref_phi[phi_idx1])/(ref_phi[phi_idx2]-ref_phi[phi_idx1]) * dy1 + ref_gain[phi_idx1][thta_idx1]
                    g2 = (i-ref_phi[phi_idx1])/(ref_phi[phi_idx2]-ref_phi[phi_idx1]) * dy2 + ref_gain[phi_idx1][thta_idx2]

                dg = g2-g1
                if dg == 0: #dg will only be zero if thta_idx1 & thta_idx2 are the same
                    gain_inter = g1
                else:
                    gain_inter = (data_theta[j]-ref_theta[thta_idx1])/(ref_theta[thta_idx2]-ref_theta[thta_idx1]) * dg + g1

                #NOW DO THE CALCULATION
                plot_gain[i][j] = gain_inter - 3.28 - 0.68 + plot_gain[i][j] + 3.25 #plot_gain is already data_delta
           # else:
                #plot_gain[i][j] = -300

    # Average multiple contours and plot on a single graph
    fig = plot.figure()
    fig.suptitle("Gain over Φ")
    ax1 = fig.add_subplot(111, projection='polar')
    for k in range(0, 9, 1): #0-8, stepsize = 1     #PHI from a -> a+9
        r = {}
        for i in range(k*10, k*10+10, 1):           #Step through PHI
            for j in range(len(data_theta)):        #Grab all THETAs
                if plot_gain[i][j] != None:         #Values that aren't None
                    theta = data_theta[j]* np.pi / 180
                    if theta not in r:
                        r[theta] = []
                    r[theta].append(plot_gain[i][j])

        r = np.array(sorted(r.items()))
        for m in range(len(r)):
            r[m][1] = np.mean(r[m][1])
        ax1.plot(r[:,0], r[:,1], label="Φ=" + str(k*10) + "-" + str(k*10+9), alpha=0.5)

    f = "../Exported Plots/GainPlots/" + "AntennaGainPolar_Contour.png"
    plot.legend()
    plot.savefig(f, dpi=300, orientation='landscape', bbox_inches='tight')
    fig.clear()
    plot.close(fig)


    """# Plots multiple contours on multiple graphs
    for k in range(0, 9, 1): #0-8, stepsize = 1     #PHI from a -> a+9
        fig = plot.figure()
        fig.suptitle("Gain Φ=" + str(k*10) + "-" + str(k*10+9))
        ax1 = fig.add_subplot(111, projection='polar')

        for i in range(k*10, k*10+10, 1):           #Step through PHI
            r = []
            theta = []

            #Grab only the values that aren't None
            for j in range(len(data_theta)):        #Grab all THETAs
                if plot_gain[i][j] != None:
                    r.append(plot_gain[i][j])
                    theta.append(data_theta[j]* np.pi / 180) 
            ax1.plot(theta, r)

        f = "../Exported Plots/GainPlots/" + "AntennaGainPolar" + str(k) + ".png"
        plot.savefig(f, dpi=300, orientation='landscape', bbox_inches='tight')
        fig.clear()
        plot.close(fig)
    """
    #Combine the combined contours
    fig = plot.figure()
    fig.suptitle("Gain across Φ")
    ax1 = fig.add_subplot(111, projection='polar')
    r = []   
    theta = []
    for i in range(len(data_phi)): #Step through all PHIs
        for j in range(len(data_theta)):
            if plot_gain[i][j] != None:
                r.append(plot_gain[i][j])
                theta.append(data_theta[j] * np.pi/180)
        #ax1.scatter(theta, r)
        r.clear()
        theta.clear()

    r_avg = []
    theta_avg = []
    for j in range(len(data_theta)): #Step through all THETAs 
        r_avg.append([])
        theta_avg.append(data_theta[j] * np.pi/180)
        for i in range(len(data_phi)):
            if plot_gain[i][j] != None:
                r_avg[j].append(plot_gain[i][j])
                if j == 78:
                    print(78)
        
        r_avg[j] = np.mean(r_avg[j])

    ax1.plot(theta_avg, r_avg)
    f = "../Exported Plots/GainPlots/" + "AntennaGainPolar.png"
    plot.savefig(f, dpi=300, orientation='landscape', bbox_inches='tight')
    fig.clear()
    plot.close(fig)
    
    """
    # Now plot 3D
    min_val = 0
    for i in range(len(plot_gain)):
        for j in range(len(plot_gain[i])):
            if plot_gain[i][j] != None:
                if plot_gain[i][j] < min_val:
                    min_val = plot_gain[i][j]

    count = 0
    x = []
    y = []
    z = []
    for i in range(len(plot_gain)):
        for j in range(len(plot_gain[i])):
            if plot_gain[i][j] == None:
                plot_gain[i][j] = 0
            else:
                plot_gain[i][j] = plot_gain[i][j] - min_val
                count = count + 1
                x.append(plot_gain[i][j] * np.sin(data_phi[i]) * np.cos(data_theta[j]))
                y.append(plot_gain[i][j] * np.sin(data_phi[i]) * np.sin(data_theta[j]))
                z.append(plot_gain[i][j] * np.cos(data_phi[i]))
    triang = mtri.Triangulation(x,y)

    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(triang,z, cmap = plot.get_cmap('hot'), linewidth=0.2, antialiased=True) 

    THETA, PHI = np.meshgrid(data_theta, data_phi)
    X = plot_gain * np.sin(PHI) * np.cos(THETA)
    Y = plot_gain * np.sin(PHI) * np.sin(THETA)
    Z = plot_gain * np.cos(PHI)

    for i in range(len(Z)):
        for j in range(len(Z[i])):
            if Z[i][j] == 0:
                Z[i][j] == np.nan
    print(count, len(Z)*len(Z[0]))
    
    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z)#, rstride=1, cstride=1, cmap=plot.get_cmap('jet'), linewidth=0, antialiased=False, alpha=0.5)"""
    plot.tight_layout()
    f = "../Exported Plots/GainPlots/" + "AntennaGainPolar3D.png"
    plot.show()
    plot.savefig(f, dpi=300, orientation='landscape', bbox_inches='tight')
    fig.clear()
    plot.close(fig)

#Function:
#Description:
def Plot_Surface(file_in):
    data = pickle.load(open( file_in, "rb" ))
    
    theta = data.get('theta')
    phi = data.get('phi')
    R = data.get('r')

    n = len(theta)
    m = len(phi)
    
    #Convert to radians if needed
    if theta[len(theta)-1] > (2*np.pi):
        theta = theta * np.pi / 180
        phi = phi * np.pi / 180

    #Calculate offset
    offset = None
    for i in range(n):
        for j in range(m):
            if not np.isnan(R[i][j]):
                if offset == None:
                    offset = R[i][j]
                else:
                    if R[i][j] < offset:
                        offset = R[i][j]    

    if offset < 0:      #Only need offset if there are negative numbers 
        offset = np.abs(offset)
    else:
        offset = 0
    
    #Add in the offset
    for i in range(n):
        for j in range(m):
            if not np.isnan(R[i][j]):
                R[i][j] = R[i][j] + offset

    #Plot Surface
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
    #TODO:SAVE THIS
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
        
        tmp_r = np.mean(tmp_r)                                          #Average the three values from the triangle and append to the color map for that point.
        color_map.append(tmp_r)

    # Finally plot the surface mesh
    fig = ff.create_trisurf(x=x,y=y,z=z,simplices=simp,title=file_in, colormap='Viridis', color_func=color_map)
    fig.show()

#Function:
#Description: 
def Plot_Scatter(file_in):
    data = pickle.load(open( file_in, "rb" ))
    
    theta = data.get('theta')
    phi = data.get('phi')
    R = data.get('r')

    n = len(theta)
    m = len(phi)
    
    #Convert to radians if needed
    if theta[len(theta)-1] > (2*np.pi):
        theta = theta * np.pi / 180
        phi = phi * np.pi / 180

    #Calculate offset
    offset = None
    for i in range(n):
        for j in range(m):
            if not np.isnan(R[i][j]):
                if offset == None:
                    offset = R[i][j]
                else:
                    if R[i][j] < offset:
                        offset = R[i][j]    

    if offset < 0:      #Only need offset if there are negative numbers 
        offset = np.abs(offset)
    else:
        offset = 0

    # Average Values
    data = {"phi":[],"theta":[],"r":[],"r_db":[]} #phi,theta,r
    for i in range(n):
        for j in range(m):
            if not np.isnan(R[i][j]):
                data["theta"].append(i * np.pi / 180) #i is the same as phi(i)
                data["phi"].append(j * np.pi / 180)
                data["r"].append(R[i][j] + offset)
                data["r_db"].append(R[i][j])
         
    x = data["r"] * np.sin(data["theta"]) * np.cos(data["phi"])      #sin(PHI), cos(THETA)       #WHERE THETA @ Z-AXIS = 0, @XY-PLANE = 90
    y = data["r"] * np.sin(data["theta"]) * np.sin(data["phi"])
    z = data["r"] * np.cos(data["theta"])
    fig = go.Figure(
        data = [
            go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=6, colorscale='Viridis', color=data["r_db"]))
        ]
    )
    fig.update_layout(title_text=file_in)
    #f = "../Exported Plots/" + file_in.split('.')[0] + "-3DCombinedSV.png"
    fig.show()

#Function:
#Definition:
def PlotRawData(file_in):
    #### Plot CNO Data 
    titles = []
    for key in List_By_SV:
        titles.append(str(key))
    rcount = int(len(List_By_SV)/5) + 1

    maintitle = "CNO(dB) by SV ID - " + file_in
    fig = make_subplots(rows=rcount, cols=5, subplot_titles=titles)
    i = 0
    for key in List_By_SV:
        x = np.linspace(1,len(List_By_SV[key]["cno"]),len(List_By_SV[key]["cno"]))
        fig.add_trace(
            go.Scatter(y=List_By_SV[key]["cno"],mode='lines',line=dict(color="#d32f2f")),row=int(i/5)+1,col=(i%5)+1)
        i = i + 1
    f = "../Exported Plots/" + file_in.split('.')[0] + "-CNOBySV.png"
    fig.update_layout(title_text=maintitle, showlegend=False)
    fig.update_layout(height=1440,width=2560)
    fig.show()
    
    #### Plot THETA / elevation
    maintitle = "θ by SV ID - " + file_in
    fig = make_subplots(rows=rcount, cols=5, subplot_titles=titles)
    i = 0
    for key in List_By_SV:
        x = np.linspace(1,len(List_By_SV[key]["theta"]),len(List_By_SV[key]["theta"]))
        fig.add_trace(
            go.Scatter(y=List_By_SV[key]["theta"],mode='lines',line=dict(color="#d32f2f")),row=int(i/5)+1,col=(i%5)+1)
        i = i + 1
    f = "../Exported Plots/" + file_in.split('.')[0] + "-THETABySV.png"
    fig.update_layout(title_text=maintitle, showlegend=False)
    fig.update_layout(height=1440,width=2560)
    #fig.show()

    #### Plot PHI / Azimuth
    maintitle = "Φ by SV ID - " + file_in
    fig = make_subplots(rows=rcount, cols=5, subplot_titles=titles)
    i = 0
    for key in List_By_SV:
        x = np.linspace(1,len(List_By_SV[key]["phi"]),len(List_By_SV[key]["phi"]))
        fig.add_trace(
            go.Scatter(y=List_By_SV[key]["phi"],mode='lines',line=dict(color="#d32f2f")),row=int(i/5)+1,col=(i%5)+1)
        i = i + 1
    f = "../Exported Plots/" + file_in.split('.')[0] + "-THETABySV.png"
    fig.update_layout(title_text=maintitle, showlegend=False)
    fig.update_layout(height=1440,width=2560)
    #fig.show()

##############################################################################################################
# DATA PARSING FUNCTIONS 
##############################################################################################################
# Function: GPS_Parse_SV()
# Details: Parses the SV and timestamp data out of the GPS txt file
def GPS_Parse_SV(file_name, file_out):
    parsedData = []
    my_gps = MicropyGPS()
    hasFix = False

    #Get the total number of lines in the file
    num_lines = sum(1 for line in open(file_name))

    #Open the file and start parsing
    with tqdm(total=num_lines, desc="Loading GPS Data [" + file_name + "]") as t:
        with open(file_name) as f:
            while True:
                line = f.readline()
                t.update()
                if("" == line):
                    pickle.dump(parsedData, open(file_out, "wb"))
                    print("Pulled", len(parsedData), "GPS messages")
                    break

                if(line.find("$GNGGA") != -1):
                    if(hasFix):
                        #Convert timestamp single digits to double digits
                        if my_gps.timestamp[0] < 10:
                            hour = "0" + str(my_gps.timestamp[0])
                        else:
                            hour = str(my_gps.timestamp[0])

                        if my_gps.timestamp[1] < 10:
                            minute = "0" + str(my_gps.timestamp[1])
                        else:
                            minute = str(my_gps.timestamp[1])

                        if my_gps.timestamp[2] < 10:
                            second = "0" + str(my_gps.timestamp[2])
                        else:
                            second = str(my_gps.timestamp[2])

                        timestamp = hour + minute + second
                        parsedData.append([timestamp, my_gps.satellite_data])

                    for x in line:
                        my_gps.update(x)
                    if(my_gps.fix_stat):
                        hasFix = True
                    else:
                        hasFix = False

                if(line.find("$GPGSV") != -1 or line.find("$GLGSV") != -1):
                    for x in line:
                        my_gps.update(x)

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

    num_lines = sum(1 for line in open(file_in))

    #Open the file and start parsing
    with tqdm(total=num_lines, desc="Loading Reference Data [" + file_in + "]") as t:
        with open(file_in) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            
            start = False #This indicates if the script is actively pulling data (state machine)
            row_idx = 0
            line_count = 1
            for row in csv_reader:  #Loop through all rows
                t.update()  #Increment progress bar

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
    print('Done')

##############################################################################################################
# EXTRA FUNCTIONS 
##############################################################################################################
#Function: PlotDeltaCNO()
#Description: Will plot file1-file2
def PlotDeltaCNO(file1_in, file2_in):
    data1 = pickle.load(open (file1_in, "rb"))
    data2 = pickle.load(open (file2_in, "rb"))

    #Setup Plot
    fig_size = plot.rcParams["figure.figsize"]
    fig_size[0] = 20
    fig_size[1] = 16
    plot.rcParams["figure.figsize"] = fig_size
    fig = plot.figure()
    fig.suptitle("ΔCNO - " + file1_in + "-" + file2_in)

    #Subtract and Plot
    i = 0
    for key in data1:
        if key in data2:
            i = i + 1
            ax1 = fig.add_subplot(6,6,i)
            delta = np.subtract(data1[key]["cno"], data2[key]["cno"])
            smooth = pd.Series(delta).rolling(window=100).mean()
            ax1.plot(delta)
            ax1.plot(smooth)
            ax1.set_title(str(key))

    plot.tight_layout()
    plot.subplots_adjust(wspace=0.2, hspace=0.35)
    f = "../Exported Plots/" + "Combined_DeltaCNO.png"
    plot.savefig(f, dpi=300, orientation='landscape', bbox_inches='tight')
    fig.clear()
    plot.close(fig)

#Function: PlotReference()
#Description:
def PlotReference(file_in):
    data = pickle.load(open( file_in, "rb" ))
    theta = data.get('theta')
    phi = data.get('phi')
    gain = data.get('gain')

    offset = np.abs(np.min(gain))
    for i in range(len(gain)):
        for j in range(len(gain[i])):
            gain[i][j] = gain[i][j] + offset

    #Convert to cartesian
    THETA, PHI = np.meshgrid(theta, phi)
    X = gain * np.sin(PHI) * np.cos(THETA)
    Y = gain * np.sin(PHI) * np.sin(THETA)
    Z = gain * np.cos(PHI)

    fig = go.Figure(
        go.Surface(x=X,y=Y,z=Z)
    )
    fig.show()


if __name__ == "__main__":
    main()
