import os
import datetime
import pickle
import math
import csv
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd

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
    #GPS_Parse_SV("Dipole-7-28-20.TXT", "Dipole_Time.obj")
    #GPS_Parse_SV("Turnstile-7-28-20.TXT", "Turnstile_Time.obj")
    #Parse3DFile('1580000000', 'rhcp', "ReferenceTurnstile3D.csv", "Reference3D_Data.obj")

    #### Time Align Data ####
    #MatchUpData("Dipole_Time.obj", "Turnstile_Time.obj", "CompiledData.obj")

    #### Clean up data and delete bad SVs ####
    #Only run this once due to the averaging function
    #CleanData("Dipole_Time.obj", "Dipole_SV.obj", False) 
    #CleanData("Turnstile_Time.obj", "Turnstile_SV.obj", False)
    #PlotDeltaCNO("Dipole_SV.obj", "Turnstile_SV.obj")

    #### Combine the data and prep for delta calculations ####
    #CombineData("Dipole_Time.obj", "Turnstile_Time.obj", "CompiledData.obj")
    #Calculate_2D("CompiledData.obj", "Reference2D_Data.obj")
    Calculate_3D("CompiledData.obj",  "Reference3D_Data.obj")
    #Calculate_SV_Movement("Dipole_Time.obj", None)

    print("main finished")

#Function: CleanData()
#Description: Will clean data up and plot SV data. Will also save the new data format conversion from timestamp keyed to satellite keyed
def CleanData(file_in, file_out, save_plots):
    #In order to clean the data, we need to seperate the SVs (keep timestamps), and then smooth the phi, cno, az
    data = pickle.load(open ( file_in, "rb" ))

    #Reorganizing the list by SV, not 
    List_By_SV = {}
    for time in data: #list of timestamps
        for key in data.get(time): #list of SVs by ID
            val = data[time][key][:]

            if val[0] != None and val[1] != None and val[2] != None:    #Remove empty datasets
                if key not in List_By_SV:
                    List_By_SV[key] = {}
                if "elev" not in List_By_SV[key]:
                    List_By_SV[key]["elev"] = []
                if "az" not in List_By_SV[key]:
                    List_By_SV[key]["az"] = []
                if "cno" not in List_By_SV[key]:
                    List_By_SV[key]["cno"] = []
                if "time" not in List_By_SV[key]:
                    List_By_SV[key]["time"] = []

                List_By_SV[key]["elev"].append(val[0])
                List_By_SV[key]["az"].append(val[1])
                List_By_SV[key]["cno"].append(val[2])
                List_By_SV[key]["time"].append(time)    #Add timestamp for future reference
    

    #Before plotting let's setup the plot size
    fig_size = plot.rcParams["figure.figsize"]
    fig_size[0] = 20
    fig_size[1] = 16
    plot.rcParams["figure.figsize"] = fig_size

    # Only show and save plots of requested
    if save_plots:
        # Plot CNO Data
        fig = plot.figure()
        fig.suptitle("CNO(dB) by SV ID - " + file_in)
        i = 0
        for key in List_By_SV:
            i = i + 1
            ax1 = fig.add_subplot(5,4,i)
            ax1.plot(List_By_SV[key]["cno"])
            smooth_data = pd.Series(List_By_SV[key]["cno"]).rolling(window=10).mean()
            ax1.plot(smooth_data)
            ax1.set_title(str(key))

        plot.tight_layout()
        plot.subplots_adjust(wspace=0.2, hspace=0.35)
        f = "./Exported/" + file_in.split('.')[0] + "-CNOBySV.png"
        plot.savefig(f, dpi=300, orientation='landscape', bbox_inches='tight')
        fig.clear()
        plot.close(fig)

        #Plot PHI / elevation
        fig = plot.figure()
        fig.suptitle("Φ by SV ID - " + file_in)
        i = 0
        for key in List_By_SV:
            i = i + 1
            ax1 = fig.add_subplot(5,4,i)
            ax1.plot(List_By_SV[key]["elev"])
            ax1.set_title(str(key))

        plot.tight_layout()
        plot.subplots_adjust(wspace=0.2, hspace=0.35)
        f = "./Exported/" + file_in.split('.')[0] + "-PHIBySV.png"
        plot.savefig(f, dpi=300, orientation='landscape', bbox_inches='tight')
        fig.clear()
        plot.close(fig)

        #Plot THETA / Azimuth
        fig = plot.figure()
        fig.suptitle("θ by SV ID - " + file_in)
        i = 0
        for key in List_By_SV:
            i = i + 1
            ax1 = fig.add_subplot(5,4,i)
            ax1.plot(List_By_SV[key]["az"])
            ax1.set_title(str(key))

        plot.tight_layout()
        plot.subplots_adjust(wspace=0.2, hspace=0.35)
        f = "./Exported/" + file_in.split('.')[0] + "-THETABySV.png"
        plot.savefig(f, dpi=300, orientation='landscape', bbox_inches='tight')
        fig.clear()
        plot.close(fig)

        #Plot the 3D data for each SV
        fig = plot.figure()
        fig.suptitle("3D Plot of SV - " + file_in)
        i = 0
        for key in List_By_SV:
            i = i + 1
            ax1 = fig.add_subplot(5,4,i, projection='3d')
            phi = []
            theta = []
            for j in range(len(List_By_SV[key]["elev"])):
                phi.append(List_By_SV[key]["elev"][j] * np.pi / 180)
                theta.append(List_By_SV[key]["az"][j] * np.pi / 180)

            r = np.ones(len(phi))
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            ax1.plot(x,y,z, label=str(key))
            ax1.set_title(str(key))

        plot.tight_layout()
        plot.subplots_adjust(wspace=0.2, hspace=0.35)
        f = "./Exported/" + file_in.split('.')[0] + "-3DSV.png"
        plot.savefig(f, dpi=300, orientation='landscape', bbox_inches='tight')
        fig.clear()
        plot.close(fig)


        #Plot the combined 3D data
        fig = plot.figure()
        fig.suptitle("Combined 3D SV Plot - " + file_in)
        ax1 = fig.add_subplot(111, projection='3d')
        for key in List_By_SV:
            phi = []
            theta = []
            for j in range(len(List_By_SV[key]["elev"])): #TODO FIND A BETTER WAY FOR ELEMENT WISE ARRAY MULTIPLICATION
                phi.append(List_By_SV[key]["elev"][j] * np.pi / 180)
                theta.append(List_By_SV[key]["az"][j] * np.pi / 180)

            r = np.ones(len(phi))
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            ax1.plot(x,y,z, label=str(key))
        f = "./Exported/" + file_in.split('.')[0] + "-3DCombinedSV.png"
        plot.legend()
        plot.savefig(f, dpi=300, orientation='landscape', bbox_inches='tight')
        fig.clear()
        plot.close(fig)

    #Continue cleaning the data, remove any SV datasets that are weird
    #Smooth and save data 
    for key in List_By_SV:
        smooth_data = pd.Series(List_By_SV[key]["cno"]).rolling(window=10).mean()
        val = smooth_data.values
        for i in range(len(val)):
            if str(val[i]) == "nan":
                val[i] = List_By_SV[key]["cno"][i]
        
        List_By_SV[key]["cno"] = val[:]

    #Delete any bad data you find
    #Temporarily this is done manually by SV ID
    to_delete = [46,51,15]
    for key in to_delete:
        if key in List_By_SV:
            List_By_SV.pop(key)

    #Now save the data back - both original timestamp and new satellite ID keyed
    data_to_save = {}
    for key in List_By_SV:
        for time in List_By_SV[key]["time"]:
            if time not in data_to_save:
                data_to_save[time] = {}
            if key not in data_to_save[time]:
                data_to_save[time][key] = []
            
            idx = List_By_SV[key]["time"].index(time)
            data_to_save[time][key].append(List_By_SV[key]["elev"][idx])
            data_to_save[time][key].append(List_By_SV[key]["az"][idx])
            data_to_save[time][key].append(List_By_SV[key]["cno"][idx])


    pickle.dump(List_By_SV, open(file_out, "wb"))
    pickle.dump(data_to_save, open(file_in, "wb"))
    print("Finished cleaning " + file_in)

#Function: 
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
            ax1 = fig.add_subplot(5,4,i)
            delta = np.subtract(data1[key]["cno"], data2[key]["cno"])
            smooth = pd.Series(delta).rolling(window=10).mean()
            ax1.plot(delta)
            ax1.plot(smooth)
            ax1.set_title(str(key))

    plot.tight_layout()
    plot.subplots_adjust(wspace=0.2, hspace=0.35)
    f = "./Exported/" + "Combined_DeltaCNO.png"
    plot.savefig(f, dpi=300, orientation='landscape', bbox_inches='tight')
    fig.clear()
    plot.close(fig)

def MatchUpData(file1_in, file2_in, file_out):
    data1 = pickle.load(open( file1_in, "rb" )) #Dipole
    data2 = pickle.load(open( file2_in, "rb" )) #Turnstile

    #Before aligning data, remove any missing datapoints
    remove1 = []
    remove2 = []
    for time in data1:
        for key in data1.get(time):
            val = data1[time][key]
            if val[0] == None or val[1] == None or val[2] == None:
                remove1.append([time, key])
    for time in data2:
        for key in data2.get(time):
            val = data2[time][key]
            if val[0] == None or val[1] == None or val[2] == None:
                remove2.append([time, key])

    for x in remove1:
        data1[x[0]].pop(x[1], None)
        print("..removed empty id", x[1], "at", x[0])
    for x in remove2:
        data2[x[0]].pop(x[1], None)
        print("..removed empty id", x[1], "at", x[0])
    
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
        data1.pop(x, None) #remove key from data1 dict
        print("..removed", x, "from data1")
    for x in remove2:
        data2.pop(x, None) #remove key from data2 dict
        print("..removed", x, "from data2")

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
        print("..removed id", x[1], "at", x[0])
    for x in remove2:
        data2[x[0]].pop(x[1], None)
        print("..removed id", x[1], "at", x[0])


    #Reorganizing the list by SV, not 
    List_By_SV1 = {}
    for time in data1: #list of timestamps
        for key in data1.get(time): #list of SVs by ID
            val = data1[time][key][:]

            if val[0] != None and val[1] != None and val[2] != None:    #Remove empty datasets
                if key not in List_By_SV1:
                    List_By_SV1[key] = {}
                if "elev" not in List_By_SV1[key]:
                    List_By_SV1[key]["elev"] = []
                if "az" not in List_By_SV1[key]:
                    List_By_SV1[key]["az"] = []
                if "cno" not in List_By_SV1[key]:
                    List_By_SV1[key]["cno"] = []
                if "time" not in List_By_SV1[key]:
                    List_By_SV1[key]["time"] = []

                List_By_SV1[key]["elev"].append(val[0])
                List_By_SV1[key]["az"].append(val[1])
                List_By_SV1[key]["cno"].append(val[2])
                List_By_SV1[key]["time"].append(time)    #Add timestamp for future reference
    

    #Reorganizing the list by SV, not 
    List_By_SV2 = {}
    for time in data2: #list of timestamps
        for key in data2.get(time): #list of SVs by ID
            val = data2[time][key][:]

            if val[0] != None and val[1] != None and val[2] != None:    #Remove empty datasets
                if key not in List_By_SV2:
                    List_By_SV2[key] = {}
                if "elev" not in List_By_SV2[key]:
                    List_By_SV2[key]["elev"] = []
                if "az" not in List_By_SV2[key]:
                    List_By_SV2[key]["az"] = []
                if "cno" not in List_By_SV2[key]:
                    List_By_SV2[key]["cno"] = []
                if "time" not in List_By_SV2[key]:
                    List_By_SV2[key]["time"] = []

                List_By_SV2[key]["elev"].append(val[0])
                List_By_SV2[key]["az"].append(val[1])
                List_By_SV2[key]["cno"].append(val[2])
                List_By_SV2[key]["time"].append(time)    #Add timestamp for future reference
    
    for key in List_By_SV1:
        print(key,len(List_By_SV1[key]["elev"]),len(List_By_SV2[key]["elev"]))

    pickle.dump(data1, open( file1_in, "wb"))
    pickle.dump(data2, open( file2_in, "wb"))
    print("Finished time align")

def CombineData(file1_in, file2_in, file_out):
    data1 = pickle.load(open( file1_in, "rb" )) #Dipole
    data2 = pickle.load(open( file2_in, "rb" )) #Turnstile

    # First build the 1xn & 1xm arrays for theta and phi, respectively
    # This is fuckin dumb, just map the whole range
    phi = list(np.linspace(0,90,91))
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
            
            if np.abs(elev - data2[time][key][0])>=2 or np.abs(az - data2[time][key][1])>=2:
                print("Mismatch:",elev,data2[time][key][0],az,data2[time][key][1])
            
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

    # Work the GPS data so it's ready
    for i in range(len(data_theta)):
        if data_theta[i] > 180:
            data_theta[i] = data_theta[i] - 360

    # Work with the ref data so it's ready
    offset = np.abs(ref_theta[0])
    for i in range(len(ref_theta)):
        #Skip the offset
        #ref_theta[i] = ref_theta[i] + offset
        ref_theta[i] = int(ref_theta[i] / np.pi * 180)

    for i in range(len(ref_phi)):
        ref_phi[i] = int(ref_phi[i] / np.pi * 180)

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
    #Generate a new array
    plot_gain = data_delta

    #For each data_delta value, find the index and corresponding phi/theta
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
            else:
                plot_gain[i][j] = -300


    # Plot multiple contours
    
    #ax1 = plot.subplot(121)
    #ax2 = plot.subplot(122, projection='polar')
    
    for k in range(0, 9, 1): #0-8, stepsize = 1
        fig = plot.figure()
        fig.suptitle("Gain θ=" + str(k*10) + "-" + str(k*10+9))
        ax1 = fig.add_subplot(111, projection='polar')

        for i in range(k*10, k*10+10, 1):
            r = []
            theta = []

            #Grab only the values that aren't None
            for j in range(len(data_theta)):
                if plot_gain[i][j] != -300:
                    r.append(plot_gain[i][j])
                    theta.append(data_theta[j]* np.pi / 180) 

            #Sort
            sorted_t = np.sort(theta)
            sorted_r = []
            for l in sorted_t:
                sorted_r.append(r[theta.index(l)])

            
            ax1.plot(sorted_t, sorted_r)


        f = "./Exported/GainPlots/" + "AntennaGainPolar" + str(k) + ".png"
        plot.savefig(f, dpi=300, orientation='landscape', bbox_inches='tight')
        fig.clear()
        plot.close(fig)

    # Now plot 3D
    #Clean up values
    min_val = 0
    for i in range(len(plot_gain)):
        for j in range(len(plot_gain[i])):
            if plot_gain[i][j] < min_val and plot_gain[i][j] != -300:
                min_val = plot_gain[i][j]

    count = 0
    for i in range(len(plot_gain)):
        for j in range(len(plot_gain[i])):
            if plot_gain[i][j] == -300:
                plot_gain[i][j] = 0
            else:
                plot_gain[i][j] = plot_gain[i][j] - min_val
                count = count + 1

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
    ax.scatter(X, Y, Z)#, rstride=1, cstride=1, cmap=plot.get_cmap('jet'), linewidth=0, antialiased=False, alpha=0.5)
    plot.tight_layout()
    f = "./Exported/GainPlots/" + "AntennaGainPolar3D.png"
    plot.savefig(f, dpi=300, orientation='landscape', bbox_inches='tight')
    fig.clear()
    plot.close(fig)



####################################### DEPRECATED FUNCTIONS ###################################
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