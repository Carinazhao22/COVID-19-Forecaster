# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
!pip install opencage
from google.cloud import bigquery
import math
from opencage.geocoder import OpenCageGeocode
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

data = pd.read_csv('/kaggle/input/update/covid_19_data.csv')
data = data.rename(columns={"Province/State": "State",
                            "Country/Region": "Country", "ObservationDate": "Date"})
# remove data that cannot find location in API
data = data.loc[data['Country'] != 'Others']
# data=data.iloc[:10000]
data['lat'] = ""
data['lon'] = ""
data


 
# get api key from:  https://opencagedata.com
key = '2627efc84a49443cbb7bdeb28c491330'
geocoder = OpenCageGeocode(key)


def isnan(value):
    try:
        import math
        return math.isnan(float(value))
    except:
        return False


list_lat = []   # create empty lists
list_long = []
store = []
store_lat = []
store_long = []
for index, row in data.iterrows():  # iterate over rows in dataframe
    City = row['State']
    State = row['Country']
    if isnan(City) == False:
        query = str(City)+','+str(State)
    else:
        query = str(State)

    if query in store:
        index = store.index(query)
        lat = store_lat[index]
        long = store_long[index]
    else:
        store.append(query)
        results = geocoder.geocode(query)
        lat = results[0]['geometry']['lat']
        long = results[0]['geometry']['lng']
        store_lat.append(lat)
        store_long.append(long)
    list_lat.append(lat)
    list_long.append(long)
# create new columns from lists

data['lat'] = list_lat
data['lon'] = list_long
# output
data.to_csv('complete_data.csv', index=False)
data = pd.read_csv('/kaggle/input/complete/complete_data.csv')
data  # change date type manually in Excel


mo = data['Date'].apply(lambda x: x[5:7])
da = data['Date'].apply(lambda x: x[8:10])
data['day_from_jan_first'] = (da.apply(int)
                              + 31*(mo == '02')
                              + 60*(mo == '03')
                              + 91*(mo == '04')
                              + 121*(mo == '05')
                              + 152*(mo == '06')
                              + 182*(mo == '07')
                              + 213*(mo == '08')
                              )
data


# Set your own project id here
PROJECT_ID = 'cmpt459'
client = bigquery.Client(project=PROJECT_ID)


# reference from https://www.kaggle.com/concealberti/covid-week1-weather
table1_stations = bigquery.TableReference.from_string(
    "bigquery-public-data.noaa_gsod.stations"
)

dataframe_stations = client.list_rows(
    table1_stations,
    selected_fields=[
        # station number, world metherorological org
        bigquery.SchemaField("usaf", "STRING"),
        # wban number, weather bureau army
        bigquery.SchemaField("wban", "STRING"),
        bigquery.SchemaField("country", "STRING"),
        bigquery.SchemaField("lat", "FLOAT"),
        bigquery.SchemaField("lon", "FLOAT"),
    ],
).to_dataframe()
# datatype is change after saved
# dataframe_stations.to_csv('dataframe_stations.csv', index=False)
dataframe_stations


# reference from https://www.kaggle.com/concealberti/covid-week1-weather
table1_gsod2020 = bigquery.TableReference.from_string(
    "bigquery-public-data.noaa_gsod.gsod2020"
)

dataframe_gsod2020 = client.list_rows(table1_gsod2020,
                                      selected_fields=[
                                          bigquery.SchemaField(
                                              "stn", "STRING"),  # station number
                                          bigquery.SchemaField(
                                              "wban", "STRING"),  # station number
                                          bigquery.SchemaField(
                                              "year", "INTEGER"),
                                          bigquery.SchemaField(
                                              "mo", "INTEGER"),
                                          bigquery.SchemaField(
                                              "da", "INTEGER"),
                                          # mean temp of the day
                                          bigquery.SchemaField(
                                              "temp", "FLOAT"),
                                          bigquery.SchemaField(
                                              "dewp", "FLOAT"),  # mean_dew_point
                                          # mean_sealevel_pressure
                                          bigquery.SchemaField("slp", "FLOAT"),
                                          bigquery.SchemaField(
                                              "wdsp", "FLOAT"),  # mean_wind_speed
                                          # total_precipitation
                                          bigquery.SchemaField(
                                              "prcp", "FLOAT"),
                                          bigquery.SchemaField(
                                              "sndp", "FLOAT"),  # snow_depth
                                      ],).to_dataframe()

# datatype is change after saved
# dataframe_gsod2020.to_csv('dataframe_gsod2020.csv', index=False)
dataframe_gsod2020

stations_df = dataframe_stations
twenty_twenty_df = dataframe_gsod2020


# reference from https://www.kaggle.com/concealberti/covid-week1-weather
stations_df['STN'] = stations_df['usaf'] + '-' + stations_df['wban']
twenty_twenty_df['STN'] = twenty_twenty_df['stn'] + \
    '-' + twenty_twenty_df['wban']
cols_1 = list(twenty_twenty_df.columns)
cols_2 = list(stations_df.columns)
weather_df = twenty_twenty_df[cols_1].join(stations_df[cols_2].set_index(
    'STN'), on='STN',  how='left', lsuffix='_left', rsuffix='_right')

weather_df['temp'] = weather_df['temp'].apply(
    lambda x: np.nan if x == 9999.9 else x)
weather_df['slp'] = weather_df['slp'].apply(
    lambda x: np.nan if x == 9999.9 else x)
weather_df['dewp'] = weather_df['dewp'].apply(
    lambda x: np.nan if x == 9999.9 else x)
weather_df['wdsp'] = weather_df['wdsp'].apply(
    lambda x: np.nan if x == 999.9 else x)
weather_df['prcp'] = weather_df['prcp'].apply(
    lambda x: np.nan if x == 999.9 else x)
weather_df['sndp'] = weather_df['sndp'].apply(
    lambda x: np.nan if x == 999.9 else x)

# convert everything into celsius
temp = (weather_df['temp'] - 32) / 1.8
dewp = (weather_df['dewp'] - 32) / 1.8

# compute relative humidity as ratio between actual vapour pressure (computed from dewpoint temperature)
# and saturation vapour pressure (computed from temperature) (the constant 6.1121 cancels out)
weather_df['rh'] = (np.exp((18.678*dewp)/(257.14+dewp)) /
                    np.exp((18.678*temp)/(257.14+temp)))

# calculate actual vapour pressure (in pascals)
# then use it to compute absolute humidity from the gas law of vapour
# (ah = mass / volume = pressure / (constant * temperature))
weather_df['ah'] = ((np.exp((18.678*dewp)/(257.14+dewp)))
                    * 6.1121 * 100) / (461.5 * temp)


weather_df['month'] = weather_df['mo']
weather_df['day'] = weather_df['da']
weather_df['Date'] = pd.to_datetime(weather_df[['year', 'month', 'day']])
weather_df['Date2'] = weather_df['Date']
weather_df['Date2'] = weather_df['Date2'].astype('str')
mo2 = weather_df['Date2'].apply(lambda x: x[5:7])
da2 = weather_df['Date2'].apply(lambda x: x[8:10])
weather_df['day_from_jan_first'] = (da2.apply(int)
                                    + 31*(mo2 == '02')
                                    + 60*(mo2 == '03')
                                    + 91*(mo2 == '04')
                                    + 121*(mo2 == '05')
                                    + 152*(mo2 == '06')
                                    + 182*(mo2 == '07')
                                    + 213*(mo2 == '08')
                                    )


weather_df = weather_df.dropna(subset=['lat', 'lon'])
weather_df = weather_df.reset_index(drop=True)
data = data.dropna(subset=['lat', 'lon'])
data = data.reset_index(drop=True)
weather_df.lon = weather_df.lon.astype(int)
weather_df.lat = weather_df.lat.astype(int)
data.lon = data.lon.astype(int)
data.lat = data.lat.astype(int)

CovidWeather = data.merge(
    weather_df, on=['lat', 'lon', 'day_from_jan_first'], how='left')
CovidWeather.to_csv('CovidWeather.csv', index=False)
CovidWeather
