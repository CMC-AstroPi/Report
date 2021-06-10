"""
INCLUSION OF LIBRARIES
"""
# I import this library to know the date and time
import datetime
# -------------------------------------------------------

# I import this library to set the paths
from pathlib import Path
# -------------------------------------------------------

# I import this library to put the delays
from time import sleep
# -------------------------------------------------------

# I import this library for read a old csv file
import csv
# -------------------------------------------------------

# I import this library for make request to a website
import requests
# -------------------------------------------------------

# I import this library to allow me to obtain the coordinates of the ISS
import reverse_geocoder as rg
from ephem import readtle, degree
# -------------------------------------------------------

# I import these libraries to log
import logging
import logzero
# -------------------------------------------------------

# I import this library cv
import cv2 as cv
# -------------------------------------------------------

# I import this library to do operations
import numpy as np
# -------------------------------------------------------

# I import this library for make machine learning
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
import joblib 
# -------------------------------------------------------

# I import these libraries for operating system operations
import os
from os import listdir
from os.path import isfile, join
# -------------------------------------------------------

# I import this library to do the magnetic field strength calculations
import math
# -------------------------------------------------------


"""
SET VARIABLE
"""
# parameter setting for obtaining latitude and longitude
name = "ISS (ZARYA)"
line1 = "1 25544U 98067A   21048.55483337  .00000312  00000-0  13830-4 0  9998"
line2 = "2 25544  51.6435 214.4384 0002912  25.9622  82.2108 15.48966210270059"

iss = readtle(name, line1, line2)
# -------------------------------------------------------

# variabile per indicare data e ora dello scatto della foto
data_ora_scatto = ""
# -------------------------------------------------------

# setting parameters to take a photo
numberPhoto = 1
# -------------------------------------------------------

# setting the variable to obtain the absolute path 
dir_path = Path(__file__).parent.resolve()
print(str(dir_path))
# -------------------------------------------------------

"""
SET FILES
"""
# opening and creating data file
file_data = logzero.setup_logger(name='file_data', logfile=str(dir_path)+'/'+'file_data.csv') 
file_data.info(',ID_Photo,PhotoType,MagnetometerValue,Date_Time,Latitude,Longitude,Position')
# -------------------------------------------------------


"""
INIZIO CALCOLO MACHINE LEARNING
"""
def areascaling(imagefile):
    img = cv.imread(imagefile, cv.IMREAD_COLOR)
    height,width=img.shape[:2]
    start_row,start_col=int(height*0.2),int(width*0.2)
    end_row,end_col=int(height*0.8),int(width*0.8)
    cropped=img[start_row:end_row,start_col:end_col]
    gray_img = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    img_area = np.shape(gray_img)[0]*np.shape(gray_img)[1]
    ret,thresh = cv.threshold(gray_img,130,255,cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE )
    if len(contours)>0:
        contours = [c for c in contours if cv.arcLength(c,False)>50]
    else:
        contours = []
    if len(contours)>0:   
        tot_area = np.sum([cv.contourArea(cnt) for cnt in contours])
        tot_perimeter = np.sum([cv.arcLength(cnt,True) for cnt in contours])
        scaling = [np.log(cv.contourArea(cnt))/np.log(cv.arcLength(cnt,True)) for cnt in contours]
        mean_scaling = np.average(scaling)
        tot_num = len(contours)
        num_ge_11 = len([s for s in scaling if s >= 1.1])
        return [tot_perimeter/tot_num, tot_area/tot_num, mean_scaling, num_ge_11]
    else:
        return [0.0, 0.0 ,0.0 ,0.0]
# -------------------------------------------------------

# image extractions from the folder
def analyze(folder):
    files = [f for f in listdir(folder) if (isfile(join(folder, f))) and f.endswith(".jpg")]
    data_list = []
    for f in files:
        data_list.append(areascaling(join(folder,f)))
    return [list(x) for x in zip(*data_list)]
# -------------------------------------------------------

# training function for machine learning
def training():
    low_data = analyze(str(dir_path)+'/'+"immagini_per_l_allenamento_del_machine_learning/basse/")
    high_data = analyze(str(dir_path)+'/'+"immagini_per_l_allenamento_del_machine_learning/alte/")
    other_data = analyze(str(dir_path)+'/'+"immagini_per_l_allenamento_del_machine_learning/altro/")

    Xlow = np.transpose(np.array(low_data))
    ylow = np.ones(np.shape(Xlow)[0])*1
    Xhigh = np.transpose(np.array(high_data))
    yhigh = np.ones(np.shape(Xhigh)[0])*2
    Xother = np.transpose(np.array(other_data))
    yother = np.ones(np.shape(Xother)[0])*3

    Xdata = np.concatenate((Xlow, Xhigh, Xother), axis=0)
    ydata = np.concatenate((ylow, yhigh, yother), axis=0)

    clf = RandomForestClassifier(n_estimators = 100, max_depth=8)   
    clf.fit(Xdata, ydata) 
    return clf
# -------------------------------------------------------

loaded_rf=training()
"""
FINE CALCOLO MACHINE LAERNING
"""


"""
CREATE BASIC FUNCTION
"""

# function to get latitude
def getLatitude():
    iss.compute(data_ora_scatto) # anno/mese/giorno ora:minuti:secondi
    return(iss.sublat/degree)
# -------------------------------------------------------

# function to get longitude
def getLongitude():
    iss.compute(data_ora_scatto)
    return(iss.sublong/degree)
# -------------------------------------------------------

# function to get the name of the city where the ISS is located
def getPosition():
    pos = (getLatitude(), getLongitude())
    location = rg.search(pos)
    return "lat: "+str(location[0]["lat"])+"-"+"lon: "+str(location[0]["lon"])+"-"+"city-name: "+str(location[0]["name"])
# -------------------------------------------------------

# function to get date and time
def getHourAndDate():
    with open(str(dir_path)+'/'+'file_info_error.csv', 'r') as file:
        reader = csv.reader(file)
        i=0
        for row in reader:
            i+=1
            if(i==numberPhoto):
                print(i)
                print(row)
                return str(row[0]).split(" ")[1] + "|" + str(row[0]).split(" ")[2]  
# -------------------------------------------------------

# function get date
def getDate(start_index):
    hour_and_date = getHourAndDate()
    return hour_and_date[start_index:start_index+2]
# -------------------------------------------------------

# function to obtain the values ​​recorded by the magnetometer
def getMagnetometricSensor():
    lat = getLatitude()
    lon = getLongitude()

    output = requests.get(f"https://www.ngdc.noaa.gov/geomag-web/calculators/calculateIgrfwmm?lat1={lat}&lon1={lon}&model=WMM&startYear=20{getDate(0)}&startMonth={getDate(2)}&startDay={getDate(4)}&endYear=20{getDate(0)}&endMonth={getDate(2)}&endDay={getDate(4)}&resultFormat=json")
    print(f'z={output.text.split(",")[4].split(":")[1]} y={output.text.split(",")[13].split(":")[1]} x={output.text.split(",")[21].split(":")[1]}')
    return f'z={output.text.split(",")[4].split(":")[1]} y={output.text.split(",")[13].split(":")[1]} x={output.text.split(",")[21].split(":")[1]}'
# -------------------------------------------------------

# function to convert images into data for machine learning. 
def calculate_areascaling(imagefile):
    """
    The function recognizes clouds applying a threshold to the grayscale image.
    Returns a list that contains the average perimeter of the clouds, the average area of ​​the clouds,
    the average size of a cloud and the number of clouds above a certain size (1.1)
    """
    print(imagefile)
    img = cv.imread(imagefile, cv.IMREAD_COLOR)
    height,width=img.shape[:2]
    start_row,start_col=int(height*0.2),int(width*0.2)
    end_row,end_col=int(height*0.8),int(width*0.8)
    cropped=img[start_row:end_row,start_col:end_col]
    gray_img = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    img_area = np.shape(gray_img)[0]*np.shape(gray_img)[1]
    ret,thresh = cv.threshold(gray_img,130,255,cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE )
    if len(contours)>0:
        contours = [c for c in contours if cv.arcLength(c,False)>50]
    else:
        contours = []
    if len(contours)>0:   
        tot_area = np.sum([cv.contourArea(cnt) for cnt in contours])
        tot_perimeter = np.sum([cv.arcLength(cnt,True) for cnt in contours])
        scaling = [np.log(cv.contourArea(cnt))/np.log(cv.arcLength(cnt,True)) for cnt in contours]
        mean_scaling = np.average(scaling)
        tot_num = len(contours)
        num_ge_11 = len([s for s in scaling if s >= 1.1])
        return [tot_perimeter/tot_num, tot_area/tot_num, mean_scaling, num_ge_11]
    else:
        return [0.0, 0.0 ,0.0 ,0.0]
# -------------------------------------------------------

# image extractions from the folder
def analyze_images(folder):
    global numberPhoto
    global data_ora_scatto
    
    hour = getHourAndDate().split('|')[1].split(":") # estraggo l'ora
    delta_precedente = datetime.timedelta(hours=int(hour[0]), minutes=int(hour[1]), seconds=int(hour[2]))

    files = riordinamentoArray(folder)

    print(files)

    for f in files:

        nameUltimatePhoto = f

        data_photo = calculate_areascaling(folder+nameUltimatePhoto) 

        value = machineLearning(data_photo)[0]

        if(value == 1.):
            new_name = 'low_clouds'
        elif(value==2.):
            new_name = 'high_clouds'
        else:
            new_name = 'other'

        os.rename(folder+nameUltimatePhoto, folder+new_name+nameUltimatePhoto)

        data_ora_scatto = f"20{getDate(0)}/{getDate(2)}/{getDate(4)} {getHourAndDate().split('|')[1]}"
        print(data_ora_scatto)

        saveData(new_name)

        numberPhoto+=1
# -------------------------------------------------------

# function to save data in the csv file
def saveData(photo_type):
    file_data.info(', %d, %s, %s, %s, %f, %f, %s', numberPhoto, photo_type, getMagnetometricSensor(), getHourAndDate(), getLatitude(), getLongitude(), getPosition())
# -------------------------------------------------------

# function to understand if the image contains low clouds
def machineLearning(img):
    lista = []
    lista.append(img)
    result = loaded_rf.predict(lista)
    return result
# -------------------------------------------------------

# main function
def run():
    analyze_images(str(dir_path)+"/"+"immagini_scattate_dall_AstroPi"+"/")      
# -------------------------------------------------------


def riordinamentoArray(folder):

    lista = [f for f in listdir(folder) if (isfile(join(folder, f))) and f.endswith(".jpeg")]
    
    for elemento in lista:
        el = elemento.split("_")[2].split(".")[0]
        os.rename(folder+elemento, folder+"_img_"+n_zeri(int(el))+".jpeg")

    lista = [f for f in listdir(folder) if (isfile(join(folder, f))) and f.endswith(".jpeg")]
    lista.sort()
    return lista
    

def n_zeri(n):
    val = str(n)
    if n < 10:
        val = "00"+str(n)
    elif n < 100:
        val = "0"+str(n)
    return val

run()