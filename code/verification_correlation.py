"""
File for creating the machine learning model.
"""


"""
IMPORT LIBRARIES
"""

import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from pathlib import Path
import csv
import math
import pandas as pd
import geopandas
from matplotlib import cm, colors
import pandas

"""
FUNCTION CREATION
"""

# funzione per estrarre i dati dalle nuvole
def estrarreDatiNuvoleCMC():

    tipo = []
    num = []
    lat = []
    lon = []
    mag = []

    dir_path = Path(__file__).parent.resolve()

    with open(str(dir_path)+'/'+'file_data.csv', 'r') as file:
        reader = csv.reader(file)
        
        next(reader)

        for row in reader:
            # return {"tipo":tipo, "num":num, "lat":lat, "lon":lon, "mag":mag}

            magn = math.sqrt(math.pow(float(row[3].split(" ")[2]), 2) + math.pow(float(row[3].split(" ")[4]), 2) + math.pow(float(row[3].split(" ")[6]), 2))

            tipo.append(row[2].split("_")[0].replace(" ", ""))
            num.append(row[1].replace(" ", ""))
            lat.append(row[5].replace(" ", ""))
            lon.append(row[6].replace(" ", ""))
            mag.append(magn/1000)
    
    return {"tipo":tipo, "num":num, "lat":lat, "lon":lon, "mag":mag}
# -------------------------------------------------------

# funzione per estrarre i dati dalle nuvole
def estrarreDatiNuvoleMAYWAY():
    dir_path = Path(__file__).parent.resolve()
    folder = str(dir_path)+"/immagini_myway/"
    
    tipo = []
    num = []
    lat = []
    lon = []
    mag = []

    with open(str(dir_path)+'/'+'id_foto_myway.csv', 'r') as file:
        reader_id = csv.reader(file)

        for row in reader_id:
            if(row[1].split(":")[1] == " Saved secondary photos"):
                lista_foto = [f for f in listdir(folder) if (isfile(join(folder, f))) and f.endswith(".jpg")]
                for foto in lista_foto:
                    if(foto.split("_")[0] == "other"):
                        if(foto.split("_")[1] == "secondary"):
                            if int(foto.split("_")[3].split(".")[0]) == int(row[1].split(":")[2]):
                                with open(str(dir_path)+'/'+'dati_foto_myway.csv', 'r') as file:
                                    reader_dati = csv.reader(file)
                                    for row_data in reader_dati:
                                        if(row_data[1] == row[0].split("info_logger -")[1]):
                                            magn = math.sqrt(math.pow(float(row_data[18]), 2) + math.pow(float(row_data[19]), 2) + math.pow(float(row_data[20]), 2))

                                            tipo.append("other")
                                            num.append("secondary_"+foto.split("_img_")[1].split(".")[0])
                                            lat.append(row_data[15])
                                            lon.append(row_data[16])
                                            mag.append(magn)
                                            
                    
                    else:
                        if(foto.split("_")[2] == "secondary"):
                            if int(foto.split("_")[4].split(".")[0]) == int(row[1].split(":")[2]):
                                with open(str(dir_path)+'/'+'dati_foto_myway.csv', 'r') as file:
                                    reader_dati = csv.reader(file)      
                                    for row_data in reader_dati:
                                        if(row_data[1] == row[0].split("info_logger -")[1]):
                                            magn = math.sqrt(math.pow(float(row_data[18]), 2) + math.pow(float(row_data[19]), 2) + math.pow(float(row_data[20]), 2))

                                            tipo.append(foto.split("_secondary_")[0].replace(" ", "").split("_")[0])
                                            num.append("secondary_"+foto.split("_img_")[1].split(".")[0].replace(" ", ""))
                                            lat.append(row_data[15].replace(" ", ""))
                                            lon.append(row_data[16].replace(" ", ""))
                                            mag.append(magn)


            elif(row[1].split(":")[1] == " Saved ndvi photos"):
                lista_foto = [f for f in listdir(folder) if (isfile(join(folder, f))) and f.endswith(".jpg")]
                for foto in lista_foto:
                    if(foto.split("_")[0] == "other"):
                        if(foto.split("_")[1] == "img"):
                            if int(foto.split("_")[2].split(".")[0]) == int(row[1].split(":")[2]):
                                with open(str(dir_path)+'/'+'dati_foto_myway.csv', 'r') as file:
                                    reader_dati = csv.reader(file)
                                    for row_data in reader_dati:
                                        if(row_data[1]  == row[0].split("info_logger -")[1]):
                                            magn = math.sqrt(math.pow(float(row_data[18]), 2) + math.pow(float(row_data[19]), 2) + math.pow(float(row_data[20]), 2))
                                            
                                            tipo.append("other")
                                            num.append(foto.split("_img_")[1].split(".")[0].replace(" ", ""))
                                            lat.append(row_data[15].replace(" ", ""))
                                            lon.append(row_data[16].replace(" ", ""))
                                            mag.append(magn)


                    else:
                        if(foto.split("_")[2] == "img"):
                            if int(foto.split("_")[3].split(".")[0]) == int(row[1].split(":")[2]):
                                with open(str(dir_path)+'/'+'dati_foto_myway.csv', 'r') as file:
                                    reader_dati = csv.reader(file)
                                    for row_data in reader_dati:
                                        if(row_data[1] == row[0].split("info_logger -")[1]):
                                            magn = math.sqrt(math.pow(float(row_data[18]), 2) + math.pow(float(row_data[19]), 2) + math.pow(float(row_data[20]), 2))
                                            
                                            tipo.append(foto.split("_img_")[0].replace(" ", "").split("_")[0])
                                            num.append(foto.split("_img_")[1].split(".")[0].replace(" ", ""))
                                            lat.append(row_data[15].replace(" ", ""))
                                            lon.append(row_data[16].replace(" ", ""))
                                            mag.append(magn)
                
                
                # return {"tipo":tipo, "num":num, "lat":lat, "lon":lon, "mag":mag}
    
    return {"tipo":tipo, "num":num, "lat":lat, "lon":lon, "mag":mag}
# -------------------------------------------------------

# main function for machine learning
def main():
    
    dati_cmc = estrarreDatiNuvoleCMC()
    dati_myway = estrarreDatiNuvoleMAYWAY()

    dati_cmc = creazioneGrafico(dati_cmc)
    dati_myway = creazioneGrafico(dati_myway)

    df_cmc = pandas.DataFrame(dati_cmc)
    df_myway = pandas.DataFrame(dati_myway)

    graficoPlanisfero(df_cmc)
    graficoPlanisfero(df_myway)

    plt.show()    
# -------------------------------------------------------

def creazioneGrafico(dizionario_dati):

    high_lat = []
    high_campo_magn = []

    low_lat = []
    low_campo_magn = []

    other_lat = []
    other_campo_magn = []

    nuvole_interessanti = []

    colori = []

    for tipo, num, lat, lon, mag in zip(dizionario_dati["tipo"], dizionario_dati["num"], dizionario_dati["lat"], dizionario_dati["lon"], dizionario_dati["mag"]):

        if(-60<float(lat)<-25 and -100<float(lon)<0):
            nuvole_interessanti.append(num)

        if(tipo == "high"):
            colori.append("green")
            high_lat.append(float(lat))
            high_campo_magn.append(float(mag))

        elif(tipo == "low"):
            colori.append("red")
            low_lat.append(float(lat))
            low_campo_magn.append(float(mag))
            
        else:
            colori.append("blue")
            other_lat.append(float(lat))
            other_campo_magn.append(float(mag))

    print(nuvole_interessanti)

    fig, ax = plt.subplots(1, 1)
    fig.suptitle("Correlation between clouds")

    ax.plot(high_campo_magn, high_lat, 'o', markersize=10, label="High Clouds", color="green")
    ax.plot(low_campo_magn, low_lat, 'o', markersize=10, label="Low Clouds", color="red")
    ax.plot(other_campo_magn, other_lat, 'o', markersize=10, label="Other", color="blue")

    plt.grid(True)
    ax.set_ylabel("Latitude (degrees)", fontsize=25)
    ax.set_xlabel("Magnetic Field (micro tesla)", fontsize=25)
    ax.legend()

    dizionario_dati["colori"] = colori

    return dizionario_dati

def graficoPlanisfero(df):

    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df["lon"], df["lat"]))

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    ax = world.plot(color='white', edgecolor='black', figsize=(40,40))

    gdf.plot(ax=ax, marker="o", markersize=30, color = df["colori"])


if __name__ == "__main__":
    main() 
