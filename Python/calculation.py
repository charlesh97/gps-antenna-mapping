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

from dataclasses import dataclass
from micropyGPS import MicropyGPS

from parse import GPS_Parse_SV
from parse import Parse3DFile

@dataclass
class SatelliteDataPoint:
    id: int
    elevation: int
    azimuth: int
    cno: int
    timestamp: datetime.datetime

@dataclass
class SatelliteDataPointPair:
    id: int
    elevation1: int
    azimuth1: int
    cno1: int
    elevation2: int
    azimuth2: int
    cno2: int
    timestamp: str

def main():
    #### PARSE RAW DATA ####
    #GPS_Parse_SV("../Logged/Dipole-8-2-20.TXT", "Dipole_Raw_GPS.obj")
    #GPS_Parse_SV("../Logged/Patch-8-2-20.TXT", "Ref_Raw_GPS.obj")
    #Parse3DFile('1580000000', 'rhcp', "../Reference/ReferenceTurnstile3D.csv", "./ReferenceTurnstile_Data.obj")
    Parse3DFile('1580000000', 'rhcp', "../Reference/ReferencePatchAntenna.csv", "./ReferencePatch_Data.obj")
    PlotReference("ReferencePatch_Data.obj")

    #### Time Align Data, Remove Bad SVs ####
    #MatchUpData("Dipole_Raw_GPS.obj", "Ref_Raw_GPS.obj")

    #### Clean up data  ####
    #CleanData("Dipole_Raw_GPS.obj", "Dipole_Time.obj", True) 
    #CleanData("Ref_Raw_GPS.obj", "Ref_Time.obj", True)
    #PlotDeltaCNO("Dipole_SV.obj", "Ref_SV.obj")

    #### Combine the data and prep for delta calculations ####
    #CombineData("Dipole_Time.obj", "Ref_Time.obj", "CompiledData.obj")
    #Calculate_2D("CompiledData.obj", "Reference2D_Data.obj")
    #Calculate_3D("CompiledData.obj",  "ReferenceTurnstile_Data.obj")
    #Calculate_SV_Movement("Dipole_Time.obj", None)

    print("main finished")

#Function: MatchUpData()
#Description: Time aligns data
def MatchUpData(file1_in, file2_in):
    data1 = pickle.load(open( file1_in, "rb" )) #Dipole
    data2 = pickle.load(open( file2_in, "rb" )) #Turnstile

    ####### BEFORE DATA ALIGN #########
    # Remove any missing datapoints
    # Adjust the GPS elevation and azimuth to match spherical coordinates phi and theta
    ###################################
    # Spherical Coordinates Convention Used
    #         PHI=0
    #           │      / 
    #           │    /
    #           │  /
    #           │/
    # ----------│------------ PHI=90, THETA=90
    #         / │
    #       /   │
    #     /     │
    #   PHI=90
    #   THETA=0
    ###################################
    remove1 = []
    remove2 = []
    for time in data1: #By timestamp
        for key in data1.get(time): #By satellite ID
            data1[time][key] = list(data1[time][key])   #<----- Sneak a tuple to list conversion here for later
            val = data1[time][key]
            if val[0] == None or val[1] == None or val[2] == None:  #Remove bad data
                remove1.append([time, key])


    for time in data2:
        for key in data2.get(time):
            data2[time][key] = list(data2[time][key])   #<----- Sneak a tuple to list conversion here for later
            val = data2[time][key]
            if val[0] == None or val[1] == None or val[2] == None:
                remove2.append([time, key])


    for x in remove1:
        data1[x[0]].pop(x[1], None)
    for x in remove2:
        data2[x[0]].pop(x[1], None)
    
    print("Removed " + str(len(remove1) + len(remove2)) + " empty satellite datapoints")

    #Find timestamp differences
    remove1.clear()
    remove2.clear()
    for key in data1:
        if key not in data2:
            remove1.append(key)
    for key in data2:
        if key not in data1:
            remove2.append(key)

    #Remove missing timestamps
    for x in remove1:
        data1.pop(x, None) 
    for x in remove2:
        data2.pop(x, None)

    print("Removed " + str(len(remove1) + len(remove2)) + " missing timestamps")

    #Find satellite differences
    remove1.clear()
    remove2.clear()
    for time in data1:
        for key in data1.get(time):
            if key not in data2.get(time):
                remove1.append([time, key])
    
    for time in data2:
        for key in data2.get(time):
            if key not in data1.get(time):
                remove2.append([time, key])
    
    #Remove missing satellite datapoints
    for x in remove1:
        data1[x[0]].pop(x[1], None)
    for x in remove2:
        data2[x[0]].pop(x[1], None)

    print("Removed " + str(len(remove1) + len(remove2)) + " missing satellite datapoints")

    #Check that the length of timestamps are the same
    if len(data1) != len(data2):
        print("Data1:" + str(len(data1)) + " Data2:" + str(len(data2)) + " mismatch!")
        exit()
    
    #Check that the number of satellite datapoints are the same
    total_count = 0
    length1 = {}
    for time in data1:
        for key in data1[time]:
            total_count = total_count + 1
            if key not in length1:
                length1[key] = 0
            else:
                length1[key] = length1[key] + 1
    length2 = {}
    for time in data2:
        for key in data2[time]:
            if key not in length2:
                length2[key] = 0
            else:
                length2[key] = length2[key] + 1

    for l in length1:
        if length1[l] != length2[l]:
            print("-------ERROR------")
            print("Misaligned datasets found at SV:" + str(l) + ", " + str(length1[l]) + ", " + str(length2[l]))
            exit()
        print("SV:" + str(l) + "\t-\t" + str(length1[l]) + ", " + str(length2[l]))
    
    print(str(total_count) + " individual datapoints, " + str(len(length1)) + " SVs")
    print("Data alignment check passed")

    pickle.dump(data1, open( file1_in, "wb"))
    pickle.dump(data2, open( file2_in, "wb"))
    print("Finished time align")

#Function: CleanData()
#Description: Will clean data up and plot SV data. Will also save the new data format conversion from timestamp keyed to satellite keyed
def CleanData(file_in, file_out, save_plots):
    #In order to clean the data, we need to seperate the SVs (keep timestamps), and then smooth the phi, cno, az
    data = pickle.load(open ( file_in, "rb" ))

    #First adjust the elev/azimuth values to match spherical coordinate system
    for time in data:
        for key in data.get(time):
            val = data[time][key]
            val[0] = 90 - val[0] #Convert to elev to PHI
            if val[0] > 90:
                val[0] = 90

            val[1] = 90 - val[1] #Azimuth to THETA
            if val[1] < 0:
                val[1] = val[1] + 360

    #Reorganizing the list by SV, not time
    List_By_SV = {}
    i = 0
    for time in data: #list of time
        for key in data.get(time): #list of SVs by ID
            val = data[time][key][:]
        
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

            List_By_SV[key]["phi"].append(val[0])
            List_By_SV[key]["theta"].append(val[1])
            List_By_SV[key]["cno"].append(val[2])
            List_By_SV[key]["time"].append(time)    
            List_By_SV[key]["time_idx"].append(i)   #Add timestamp(index) for future reference
        i = i + 1

    #Each key now has lists -> convert to arrays
    for key in List_By_SV:
        List_By_SV[key]["phi"] = np.array(List_By_SV[key]["phi"])   #Convert to array
        List_By_SV[key]["theta"] = np.array(List_By_SV[key]["theta"])
        List_By_SV[key]["cno"] = np.array(List_By_SV[key]["cno"])

    #Continue cleaning the data
    #Smooth and save data 
    for key in List_By_SV:
        smooth_data = pd.Series(List_By_SV[key]["cno"]).rolling(window=100).mean()
        val = smooth_data.values
        for i in range(len(val)):
            if str(val[i]) == "nan":
                val[i] = List_By_SV[key]["cno"][i]
        
        List_By_SV[key]["cno"] = val[:]

    # Only show and save plots of requested
    if save_plots:
        
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

        #### Plot PHI / elevation
        maintitle = "Φ by SV ID - " + file_in
        fig = make_subplots(rows=rcount, cols=5, subplot_titles=titles)
        i = 0
        for key in List_By_SV:
            x = np.linspace(1,len(List_By_SV[key]["phi"]),len(List_By_SV[key]["phi"]))
            fig.add_trace(
                go.Scatter(y=List_By_SV[key]["phi"],mode='lines',line=dict(color="#d32f2f")),row=int(i/5)+1,col=(i%5)+1)
            i = i + 1
        f = "../Exported Plots/" + file_in.split('.')[0] + "-PHIBySV.png"
        fig.update_layout(title_text=maintitle, showlegend=False)
        fig.update_layout(height=1440,width=2560)
        fig.show()

        #### Plot THETA / Azimuth
        maintitle = "θ by SV ID - " + file_in
        fig = make_subplots(rows=rcount, cols=5, subplot_titles=titles)
        i = 0
        for key in List_By_SV:
            x = np.linspace(1,len(List_By_SV[key]["theta"]),len(List_By_SV[key]["theta"]))
            fig.add_trace(
                go.Scatter(y=List_By_SV[key]["theta"],mode='lines',line=dict(color="#d32f2f")),row=int(i/5)+1,col=(i%5)+1)
            i = i + 1
        f = "../Exported Plots/" + file_in.split('.')[0] + "-PHIBySV.png"
        fig.update_layout(title_text=maintitle, showlegend=False)
        fig.update_layout(height=1440,width=2560)
        fig.show()
        
        #### Plot combined 3D data - Lines
        # Going to build an nxm array to store the list of R values for each SV
        # Average those values and build an unsorted list of phi/theta/r/time
        # Sort the list by time
        # Plot the lines with remaining phi/theta/r data
        maintitle = "Combined 3D SV Plot - " + file_in
        fig = go.Figure()

        i = 0
        phi_ = list(np.linspace(0,90,91))
        theta_ = list(np.linspace(0,359,360))
        n = len(phi_)
        m = len(theta_)
        combined_r = [[ [] for x in range(m)] for x in range(n)] #n x m matrix

        for key in List_By_SV:
            # Now build the R nxm array
            r = [[[[],[]] for x in range(m)] for x in range(n)] #n x m matrix
            for i in range(len(List_By_SV[key]["phi"])):
                x = List_By_SV[key]["phi"][i]
                y = List_By_SV[key]["theta"][i]
                r[x][y][0].append(List_By_SV[key]["cno"][i])
                r[x][y][1].append(List_By_SV[key]["time_idx"][i])
                combined_r[x][y].append(List_By_SV[key]["cno"][i])   #This is used in the next portion

            # Average Values
            data = {"phi":[],"theta":[],"r":[],"time":[]} #phi,theta,r,time_idx
            for i in range(n):
                for j in range(m):
                    if len(r[i][j][0]) > 0:
                        data["phi"].append(i * np.pi / 180) #i is the same as phi(i)
                        data["theta"].append(j * np.pi / 180)
                        data["r"].append(np.mean(r[i][j][0]))
                        data["time"].append(int(np.mean(r[i][j][1])))
                        
            # Use dataframe pandas to sort by time
            df = pd.DataFrame(data)
            df = df.sort_values(by='time').reset_index(drop=True)   #don't forget to reindex
            x = df["r"] * np.sin(df["phi"]) * np.cos(df["theta"])      #sin(PHI), cos(THETA)       #WHERE PHI @ Z-AXIS = 0, @XY-PLANE = 90
            y = df["r"] * np.sin(df["phi"]) * np.sin(df["theta"])
            z = df["r"] * np.cos(df["phi"])
            fig.add_trace(
                go.Scatter3d(x=x, y=y, z=z, mode='markers', name=str(key))
            )
        #f = "../Exported Plots/" + file_in.split('.')[0] + "-3DCombinedSV.png"
        fig.update_layout(title_text=maintitle)
        #fig.update_layout(height=1440,width=2560)
        fig.show()
        

        #### Plot combined 3D data - Surface
        # combined_r already has nxm (phi/theta) averaged data
        # Build x,y,z arrays
        #
        maintitle = "Combined 3D SV Plot - " + file_in
        #Build the dataset
        data = {
            "phi": [],
            "theta": [],
            "r": []
        }
        for i in range(n):
            for j in range(m):
                if len(combined_r[i][j]) > 0: #empty list otherwise
                    data["phi"].append(i)
                    data["theta"].append(j)
                    data["r"].append(np.mean(combined_r[i][j]))
        
        #Sort dataset
        pickle.dump(data, open("dataset.obj", "wb"))

        df = pd.DataFrame(data)
        df = df.sort_values(by=['phi','theta']).reset_index(drop=True)
        x = df["r"] * np.sin(df["phi"]) * np.cos(df["theta"])      #sin(PHI), cos(THETA)       #WHERE PHI @ Z-AXIS = 0, @XY-PLANE = 90
        y = df["r"] * np.sin(df["phi"]) * np.sin(df["theta"])
        z = df["r"] * np.cos(df["phi"])
        fig = go.Figure(
            #go.Mesh3d(x=x,y=y,z=z,alphahull=1)
            go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=8,color=df["r"], colorbar=dict(title="Colorbar"), colorscale='Inferno',opacity=0.8))
        )
        fig.update_layout(title_text=maintitle,coloraxis_showscale=True)
        #fig.update_layout(height=1440,width=2560)
        fig.show()

    exit()
    #Delete any bad data you find
    #Temporarily this is done manually by SV ID
    #to_delete = [46,51,15]
    #for key in to_delete:
    #    if key in List_By_SV:
    #        List_By_SV.pop(key)

    #Now save the data back - both original timestamp and new satellite ID keyed
    data_to_save = {}
    for key in List_By_SV:
        for time in List_By_SV[key]["time"]:
            if time not in data_to_save:
                data_to_save[time] = {}
            if key not in data_to_save[time]:
                data_to_save[time][key] = []
            
            idx = List_By_SV[key]["time"].index(time)
            data_to_save[time][key].append(List_By_SV[key]["phi"][idx])
            data_to_save[time][key].append(List_By_SV[key]["theta"][idx])
            data_to_save[time][key].append(List_By_SV[key]["cno"][idx])

    #pickle.dump(data_to_save, open(file_in, "wb"))
    pickle.dump(List_By_SV, open(file_out, "wb"))
    print("Finished cleaning " + file_in)


#Function: CombineData()
#Description: 
def CombineData(file1_in, file2_in, file_out):
    data1 = pickle.load(open( file1_in, "rb" )) #Dipole
    data2 = pickle.load(open( file2_in, "rb" )) #Turnstile

    # First build the 1xn & 1xm arrays for theta and phi, respectively
    # This is fuckin dumb, just map the whole range
    phi = list(np.linspace(0,89,90))
    theta = list(np.linspace(0,359,360))

    # Now build the R nxm array.
    n = len(phi)
    m = len(theta)
    r = [[None for x in range(m)] for x in range(n)] #n x m matrix

    appended = 0
    for time in data1:
        for key in data1.get(time):
            elev = data1[time][key][0] #TODO: UNCOMMENT THE PRINT MISMATCH AND FIX THE AVERAGING + 360/0 condition
            az = data1[time][key][1]
            delta = data1[time][key][2] - data2[time][key][2]
            
            #if np.abs(elev - data2[time][key][0])>=2 or np.abs(az - data2[time][key][1])>=2:
                #print("Mismatch:",elev,data2[time][key][0],az,data2[time][key][1])
            
            n_i = phi.index(elev)
            m_j = theta.index(az)

            if r[n_i][m_j] == None:
                r[n_i][m_j] = []
            
            r[n_i][m_j].append(delta)
            appended = appended + 1 

    print("Counted", appended, "datapoints")

    # Store data
    data = {}
    data['r'] = r
    data['phi'] = phi
    data['theta'] = theta
    pickle.dump(data, open(file_out, "wb"))

#Function: Calculate3D()
#Description: 
def Calculate_3D(file_in, reference_in):
    # Grab data from serialized object
    data = pickle.load(open( file_in, "rb" ))
    ref = pickle.load(open( reference_in, "rb" ))

    data_theta = data.get('theta')#once you get past 180 subtract 2pi to put you back negative radians
    data_phi = data.get('phi')
    data_delta = data.get('r')

    ref_theta = ref.get('theta')
    ref_phi = ref.get('phi')
    ref_gain = ref.get('gain')

    # Work the GPS data so it's ready  ##TODO: Where to put this section?
    for i in range(len(data_theta)):
        if data_theta[i] > 180:
            data_theta[i] = data_theta[i] - 360

    # Work with the ref data so it's ready
    offset = np.abs(ref_theta[0])
    for i in range(len(ref_theta)):
        #Skip the offset
        #ref_theta[i] = ref_theta[i] + offset
        ref_theta[i] = np.round(ref_theta[i] / np.pi * 180)

    for i in range(len(ref_phi)):
        ref_phi[i] = np.round(ref_phi[i] / np.pi * 180)

    # Average points - Each gain value in 2D array is an array of data taken from GPS.
    # Reduce to single point
    count = 0
    total_count = 0
    for i in range(len(data_phi)):
        for j in range(len(data_theta)):
            if data_delta[i][j] != None:
                total_count = total_count + len(data_delta[i][j])
                data_delta[i][j] = np.mean(data_delta[i][j])
                count = count + 1

    print("Total values:",total_count,"Unique Values:",count)

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

#Function:
#Description:
def PlotReference(file_in):
    data = pickle.load(open( file_in, "rb" ))
    theta = data.get('theta')
    phi = data.get('phi')
    gain = data.get('gain')

    #Rotate the phi around 90 degrees
    for i in range(len(phi)):
        if phi[i] - np.pi/2 < 0:
            phi[i] = np.pi/2 - phi[i]
        else:
            phi[i] = phi[i] - np.pi/2

    offset = np.abs(np.min(gain))
    for i in range(len(gain)):
        for j in range(len(gain[i])):
            gain[i][j] = gain[i][j] + offset

    THETA, PHI = np.meshgrid(theta, phi)
    """x' = x
   = r sin θ cos φ
y' = y cos α - z sin α
   = (r sin θ sin φ) cos α - (r cos θ) sin α
z' = y sin α + z cos α
   = (r sin θ sin φ) sin α + (r cos θ) cos α"""

    #Convert to cartesian
    X = gain * np.sin(PHI) * np.cos(THETA)
    Y = gain * np.sin(PHI) * np.sin(THETA)
    Z = gain * np.cos(PHI)

    #Rotate 90deg around the x-axis
    alpha = 0 * np.pi / 180
    x_ = X
    y_ = Y*np.cos(alpha) - Z*np.sin(alpha)
    z_ = Y*np.sin(alpha) + Z*np.cos(alpha)

    fig = go.Figure(
        go.Surface(x=x_,y=y_,z=z_)
    )
    fig.show()



####################################### DEPRECATED FUNCTIONS ###################################
def CSV_EXPORT():
        #with open(file_out2, mode='w', newline='') as f_out:
        #csv_writer = csv.writer(f_out, delimiter=',')
        #csv_writer.writerow(["TIMESTAMP","SV","THETA","PHI","CNO"])
        #csv_writer.writerow([float(time), key, val[0], val[1], val[2]])
    print("csv export done")

def Calculate_SV_Movement(file_in1, file_in2):
    data1 = pickle.load(open( file_in1, "rb" ))

    #Organize data by satellite ID and timestamp
    satellite1 = {}
    for time in data1: #match timestamps
        for key in data1.get(time): #subgroup of satellite SVs, key is ID
            value = data1.get(time).get(key)
            if value[0] != None and value[1] != None and value[2] != None:
                if key not in satellite1:
                    satellite1[key] = {}

                if 'phi' not in satellite1[key]:
                    satellite1[key]['phi'] = []

                if 'theta' not in satellite1[key]:
                    satellite1[key]['theta'] = []

                if 'time' not in satellite1[key]:
                    satellite1[key]['time'] = []

                satellite1[key]['phi'].append(value[0])
                satellite1[key]['theta'].append(value[1])
                satellite1[key]['time'].append(float(time))    
   
    if file_in2 != None:
        data2 = pickle.load(open( file_in2, "rb" ))
        satellite2 = {}
        for time in data2: #match timestamps
            for key in data2.get(time): #subgroup of satellite SVs, key is ID
                value = data2.get(time).get(key)
                if value[0] != None and value[1] != None and value[2] != None:
                    if key not in satellite2:
                        satellite2[key] = {}

                    if 'phi' not in satellite2[key]:
                        satellite2[key]['phi'] = []

                    if 'theta' not in satellite2[key]:
                        satellite2[key]['theta'] = []

                    if 'time' not in satellite2[key]:
                        satellite2[key]['time'] = []

                    satellite2[key]['phi'].append(value[0])
                    satellite2[key]['theta'].append(value[1])
                    satellite2[key]['time'].append(float(time))   
    print("Organized by satellite.")


    #Now print and plot this data (already sorted by timestamp)
    fig = plot.figure()
    ax1 = plot.subplot(231)
    ax2 = plot.subplot(232)
    ax3 = plot.subplot(233, projection='3d')
    ax1.set_title("Φ-"+file_in1)
    ax2.set_title("θ-"+file_in1)
    ax3.set_title("3D View-"+file_in1)

    for key in satellite1:
        sat = satellite1.get(key)
        time = (sat['time'])
        phi = []
        theta = []
        for i in range(len(time)):
            phi.append(sat['phi'][sat['time'].index(time[i])] * np.pi / 180)
            theta.append(sat['theta'][sat['time'].index(time[i])] * np.pi / 180)

        #Plot 2D
        if key == 28:
            ax1.plot(phi, label=str(key))
            ax2.plot(theta, label=str(key))
        
        #Plot 3D
        r = np.ones(len(phi))
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        ax3.plot(x,y,z, label=str(key))

        #file_name = file_in + "_ID_" + str(key) + ".png"
        #plot.savefig(file_name)
        
        #ax1.clear()
        #ax2.clear()
        #ax3.clear()

    if file_in2 != None:
        ax4 = plot.subplot(234)
        ax5 = plot.subplot(235)
        ax6 = plot.subplot(236, projection='3d')
        ax4.set_title("Φ-"+file_in2)
        ax5.set_title("θ-"+file_in2)
        ax6.set_title("3D View-"+file_in2)

        for key in satellite2:
            sat = satellite2.get(key)
            time = (sat['time'])
            phi = []
            theta = []
            for i in range(len(time)):
                phi.append(sat['phi'][sat['time'].index(time[i])] * np.pi / 180)
                theta.append(sat['theta'][sat['time'].index(time[i])] * np.pi / 180)

            #Plot 2D
            ax4.plot(phi, label=str(key))
            ax5.plot(theta, label=str(key))
            
            #Plot 3D
            r = np.ones(len(phi))
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            ax6.plot(x,y,z, label=str(key))


    plot.legend()
    plot.show()
    print('done')

def Calculate_2D(data_in, reference_in):
    # Grab data from serialized object
    data = pickle.load(open( data_in, "rb" ))
    ref = pickle.load(open( reference_in, "rb" ))

    r = data.get('r')
    phi = data.get('phi')
    theta = data.get('theta')
    n = data.get('n')
    m = data.get('m')

    data_2d = {} #Contains average
    gain = {} 
    minangle = 45

    print("Creating 2D Array...")
    for i in range(n):
        for j in range(m):
            if phi[i] < minangle:
                print("Phi too small, skipping: ", phi[i])
            if r[i][j] != None and phi[i] > minangle:
                key = str(theta[j])
                if key not in data_2d:
                    data_2d[key] = np.mean(r[i][j])
                else: 
                    data_2d[key] = (data_2d[key] + np.mean(r[i][j]))/2

    for key in data_2d:
        # Interpolate reference data
        k = int(key)
        ant1_gain = 0
        if k % 5 != 0:
            lkey = str(int(int(k)/5)*5)
            offset = (k % 5) - int(lkey)/5
            hkey = str(5 - (k % 5) + k)
            ant1_gain = (ref[hkey]-ref[lkey])/5*offset + ref[lkey]
        else:
            ant1_gain = ref[key]

        # Calculmalate antenna gain
        gain[key] = (ant1_gain - 3.28 - 0.68 + 3.25) + data_2d[key]
    
    
    thta = []
    r = []
    r_original = []
    for d in gain.keys():
        thta.append(int(d) * np.pi / 180)
        r.append(gain.get(d))
        r_original.append(data_2d.get(d))

    fig, ax = plot.subplots(2,2, subplot_kw=dict(projection='polar'))
    ax[0,0].scatter(thta, r)
    ax[0,0].set_title('Antenna Gain')
    ax[0,1].scatter(thta, r_original)
    ax[0,1].set_title('Delta Values')
    ax[1,0].plot(thta, r)
    ax[1,0].set_title('Antenna Gain')
    ax[1,1].plot(thta, r_original)
    ax[1,1].set_title('Delta Values')

    fig.suptitle('Filtered Minimum Elevation 45deg')
    plot.show()

    print("Finished 2D Plot")


if __name__ == "__main__":
    main()
