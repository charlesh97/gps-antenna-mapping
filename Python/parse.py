import os
import datetime
import pickle
import math
import csv
import matplotlib.pyplot as plot
import numpy as np

from dataclasses import dataclass
from micropyGPS import MicropyGPS

def main():
    #GPS_Parse_SV("./Logged/Dipole-7-28-20.TXT", "./Python/Dipole_Time.obj")
    #GPS_Parse_SV("./Logged/Turnstile-7-28-20.TXT", "./Python/Turnstile_Time.obj")
    #Parse2DFile("./Reference/ReferenceTurnstile2D.csv","./Python/Reference2D_Data.obj")
    #Parse3DFile('1580000000', 'rhcp', "../Reference/ReferencePatchAntenna.csv", "./Reference3D_Data.obj")


    #Plot 3D Reference File
    # Grab data from serialized object
    data = pickle.load(open( 'Reference3D_Data.obj', "rb" ))
    theta = data.get('theta')
    phi = data.get('phi')
    gain = data.get('gain')

    offset = np.abs(np.min(gain))
    for i in range(len(gain)):
        for j in range(len(gain[i])):
            gain[i][j] = gain[i][j] + offset

    offset = np.abs(theta[0])
    for i in range(len(theta)):
        theta[i] = theta[i] + offset


    # Now plot
    THETA, PHI = np.meshgrid(theta, phi)
    X = gain * np.sin(PHI) * np.cos(THETA)
    Y = gain * np.sin(PHI) * np.sin(THETA)
    Z = gain * np.cos(PHI)

    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plot.get_cmap('jet'), linewidth=0, antialiased=False, alpha=0.5)
    plot.show()     
    print("Main finished") 





# Function: Parse2DFile()
# Details: Parses the 2d polar gain information from a 2D gain CSV exported from SS3
def Parse2DFile(file_in, file_out):
    degree = {}

    with open(file_in) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0: # Don't count first line
                line_count = line_count + 1
            else:               # Grab data from other lines
                tmp = str(round(float(row[0]) + 180))
                degree[tmp] = float(row[1])

    if degree != None:
        pickle.dump(degree, open(file_out, "wb"))

# Function: Parse3DFile()
# Details: Parses the 3d polar gain information from the near field chamber csv
# Parameters:
# frequency : int of frequency e.g. 1500000000
# gain_type : str of 'lin','rhcp','lhcp'
# file_in : csv file in
# file_out : obj out
def Parse3DFile(frequency, gain_type, file_in, file_out):
    phi = []
    theta = []
    gain = []

    col_idx = -1
    col_len = 120 #120 data points in the set

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
        
        start = False
        row_idx = 0
        line_count = 1
        for row in csv_reader:  #Loop through all rows
            if line_count == 2: #Grab all the theta values
                for j in range(col_len):
                    if j == 0:
                        theta.append(float(row[j+col_idx].split('=')[1]))
                    else:
                        theta.append(float(row[j+col_idx]))

            if not start and row[0] == frequency: #Find start of frequency (row)
                start = True

            if start and row[0] == frequency:
                phi.append(float(row[1]))
                gain.append([])

                for j in range(col_len):
                    gain[row_idx].append(float(row[j+col_idx]))

                row_idx = row_idx + 1

            if start and row[0] != frequency:
                break

            line_count = line_count + 1

    data = {}
    data['theta'] = theta
    data['phi'] = phi
    data['gain'] = gain
    pickle.dump(data, open(file_out, "wb"))
    print('3D Parsing Finished')

# Function: GPS_Parse_SV()
# Details: Parses the SV and timestamp data out of the GPS txt file
def GPS_Parse_SV(file_name, file_out):
    parsedData = {}
    my_gps = MicropyGPS()
    hasFix = False

    with open(file_name) as f:
        while True:
            line = f.readline()
            if("" == line):
                pickle.dump(parsedData, open(file_out, "wb"))
                print("Pulled", len(parsedData), "GPS messages, saved to", file_out)
                break

            if(line.find("$GNGGA") != -1):
                if(hasFix):
                    #Pull all stats
                    #for key in my_gps.satellite_data:
                    #    value = my_gps.satellite_data.get(key)
                    #    parsedData.append(SatelliteDataPoint(key, value[0], value[1], value[2], my_gps.timestamp))
                    if my_gps.timestamp[2] < 10:
                        second = "0" + str(my_gps.timestamp[2])
                    else:
                        second = str(my_gps.timestamp[2])
                    key = str(my_gps.timestamp[0]) + str(my_gps.timestamp[1]) + second
                    parsedData[key] = my_gps.satellite_data

                for x in line:
                    my_gps.update(x)
                if(my_gps.fix_stat):
                    hasFix = True
                else:
                    hasFix = False

            if(line.find("$GPGSV") != -1):
                for x in line:
                    my_gps.update(x)



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


if __name__ == "__main__":
    main()