# Astro-Pi Experience Report

In this experience we verified whether there was a correlation between the strength of the Earth's magnetic field and the formation of low clouds.

## Contributing

Thanks for the help provided:
- [ESA](http://www.esa.int/) for this wonderful initiative that allowed us to identify ourselves as two space explorers.
- Our supervisor [Simone Conradi](https://github.com/conradis) for the support given.
- [ITIS Mario Delpozzo](https://www.itiscuneo.gov.it/) for the opportunity it has given us.
- [The MyWay Team](https://github.com/MyWay-AstroPi) for providing us with their data.
## Code Snippets

### Snippet 1:

This is the function that has the task of extracting 
useful data from the images.

```python
def calculateAreascaling(imagefile):
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
```

### Snippet 2:

This is the function that has the task of obtaining the
value of the Earth's magnetic field given:
- latitude
- longitude
- acquisition of date and time of the data

Thanks to the help of the site [NCEI Magnetic Field Calculators](https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#igrfwmm).

```python
def getMagnetometricSensor():
    lat = getLatitude()
    lon = getLongitude()

    output = requests.get(f"https://www.ngdc.noaa.gov/geomag-web/calculators/calculateIgrfwmm?lat1={lat}&lon1={lon}&model=WMM&startYear=20{getDate(0)}&startMonth={getDate(2)}&startDay={getDate(4)}&endYear=20{getDate(0)}&endMonth={getDate(2)}&endDay={getDate(4)}&resultFormat=json")
    print(f'z={output.text.split(",")[4].split(":")[1]} y={output.text.split(",")[13].split(":")[1]} x={output.text.split(",")[21].split(":")[1]}')
    return f'z={output.text.split(",")[4].split(":")[1]} y={output.text.split(",")[13].split(":")[1]} x={output.text.split(",")[21].split(":")[1]}'
```

### Snippet 3:

This is the function that has the purpose of creating graphics in 
the shape of a planisphere using a `pandas` dataframe.

```python
def graficoPlanisfero(df):

    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df["lon"], df["lat"]))

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    ax = world.plot(color='white', edgecolor='black', figsize=(40,40))

    gdf.plot(ax=ax, marker="o", markersize=15, color = df["colori"])
```

## Insights

For machine learning we used the [sklearn.ensemble](https://scikit-learn.org/stable/modules/ensemble.html) library and in particular the `RandomForestClassifier` module
which allowed us to recognize the type of clouds.

## Graphs

[Graphic CMC](https://github.com/CMC-AstroPi/Report/blob/main/graphs/graphic_cmc.png)

[Graphic MyWay](https://github.com/CMC-AstroPi/Report/blob/main/graphs/graphic_myway.png)

[Planisphere CMC](https://github.com/CMC-AstroPi/Report/blob/main/graphs/planisphere_cmc.png)

[Planisphere MyWay](https://github.com/CMC-AstroPi/Report/blob/main/graphs/planisphere_myway.png)

## Warning

To execute the code you need to have the images and arrange all the files in a single folder.

## Conclusion

In conclusion, based on the data returned, we can say that there is a correlation between the intensity of the Earth's magnetic field and the formation of low clouds.

For more information see the [document]().

## Authors

- [Gabriele Ferrero](https://github.com/GabrieleFerrero)
- [Isabella Bianco](https://github.com/IsabellaBianco)

  
