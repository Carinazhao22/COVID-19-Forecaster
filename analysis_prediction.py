# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

! pip install chart-studio
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import warnings
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
warnings.filterwarnings('ignore')

import re

import math
from scipy.stats import pearsonr
import statistics

from pandas.plotting import autocorrelation_plot as acplot
import matplotlib.pyplot as plt
import seaborn as sns
! pip install chart-studio
import chart_studio.plotly as py #For World Map
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.express as px       
import plotly.offline as py   
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
import matplotlib.colors as mc
import colorsys
from random import randint

import sklearn
from sklearn import metrics
from sklearn.preprocessing import StandardScaler  # For scaling dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from statsmodels.tsa.arima.model import ARIMA


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


CovidWeather = pd.read_csv('/kaggle/input/covid-weather/CovidWeather.csv')
covid_cases = CovidWeather
CovidWeather = CovidWeather.loc[CovidWeather['stn'].notnull()]
CovidWeather = CovidWeather.loc[CovidWeather['State'].notnull()]
CovidWeather = CovidWeather.applymap(
    lambda x: x.strip() if type(x) == str else x)
CityWeather = CovidWeather.sort_values(by=['State', 'Date_x'])
# CityWeather
# CovidWeather

increased_confirmed = list()
increased_Deaths = list()
increased_Recovered = list()
for index in range(len(CityWeather)):
    cur = CityWeather.iloc[index]
    if index != 0:
        pre = CityWeather.iloc[index-1]
    if index == 0:
        confirmed = int(cur['Confirmed'])
        death = int(cur['Deaths'])
        recover = int(cur['Recovered'])
        add1, add2, add3 = (confirmed, death, recover)
        increased_confirmed.append(add1)
        increased_Deaths.append(add2)
        increased_Recovered.append(add3)
    elif cur['State'] != pre['State']:
        confirmed = int(cur['Confirmed'])
        death = int(cur['Deaths'])
        recover = int(cur['Recovered'])
        add1, add2, add3 = (confirmed, death, recover)
        increased_confirmed.append(add1)
        increased_Deaths.append(add2)
        increased_Recovered.append(add3)
    else:
        confirmed1 = int(cur['Confirmed'])
        death1 = int(cur['Deaths'])
        recover1 = int(cur['Recovered'])
        confirmed = int(pre['Confirmed'])
        death = int(pre['Deaths'])
        recover = int(pre['Recovered'])
        add1 = confirmed1-confirmed
        add2 = death1-death
        add3 = recover1-recover
        if add1 < 0:
            add1 = 0
        if add2 < 0:
            add2 = 0
        if add3 < 0:
            add3 = 0
        increased_confirmed.append(add1)
        increased_Deaths.append(add2)
        increased_Recovered.append(add3)
CityWeather['increased_confirmed'] = increased_confirmed
CityWeather['increased_deaths'] = increased_Deaths
CityWeather['increased_recovered'] = increased_Recovered
# CityWeather

''''
lat: Latitude
lon: Longitude
temp: mean temperature of the day, 
dewp: mean dewpoint, 
slp: mean sealevel pressure, 
wdsp: mean wind speed, 
prcp: total precipitation, 
sndp: snow depth
rh: humidity 
'''
print('Enter 1 for viewing weather-covid19 relationship in a city.\nEnter 2 for viewing weather-covid19 relationship in a country.\nEnter 3 for viewing weather-covid19 relationship in whole world.')
query = input()

columns_X = ["lat", "lon", "temp", "dewp",
             "slp", "wdsp", "prcp", "sndp", "rh", "ah"]
columns_y = ["Confirmed", "Deaths", "Recovered"]
Country = set(CityWeather['Country'])
Country = sorted(Country)
if query == '1':
    print(Country)
    print('You can enter anything names shown above.')
    region = input()
    while region not in Country:
        print('Not Found!.You can enter anything names shown above')
        region = input()
    select_Data = CityWeather.loc[CityWeather['Country'] == region]
    all_City = set(select_Data['State'])
    print(sorted(all_City))
    print('You can enter anything shown above')
    city = input()
    while city not in all_City:
        print('Not Found! You can enter anything shown above')
        city = input()

    select_city = select_Data.loc[select_Data['State'] == city]
    # filter out extreme cases
    # select_city=select_city.loc[select_city['increased_confirmed']!=max(select_city['increased_confirmed'])]
    # select_city=select_city.loc[select_city['increased_deaths']!=max(select_city['increased_deaths'])]
    # select_city=select_city.loc[select_city['increased_recovered']!=max(select_city['increased_recovered'])]

    plt.figure(figsize=(20, 10))
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex=True)
    fig.suptitle(
        'Weather Situation. Mean Temperature,Dewpoint,Sealevel Pressure, Wind speed, Precipitation,Humidity')
    ax1.plot(select_city['day_from_jan_first'], select_city['temp'], 'black')
    ax2.plot(select_city['day_from_jan_first'], select_city['dewp'], 'blue')
    ax3.plot(select_city['day_from_jan_first'], select_city['slp'], 'gray')
    ax4.plot(select_city['day_from_jan_first'], select_city['wdsp'], 'red')
    ax5.plot(select_city['day_from_jan_first'], select_city['prcp'], 'green')
    ax6.plot(select_city['day_from_jan_first'], select_city['rh'], 'purple')
    plt.xlabel("day_from_jan_first")
    plt.show()
    plt.figure(figsize=(50, 20))
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharex=True, sharey=True)
    fig.suptitle('Covid Situation.Increased Confirmed,Deaths and Recovered')
    ax1.plot(select_city['day_from_jan_first'],
             select_city['increased_confirmed'], 'b*')
    ax2.plot(select_city['day_from_jan_first'],
             select_city['increased_deaths'], 'r*')
    ax3.plot(select_city['day_from_jan_first'],
             select_city['increased_recovered'], 'g*')
    plt.xlabel("day_from_jan_first")
    plt.ylabel('Number of cases')
    plt.show()

confirmed = [list() for _ in
             range(2)]
death = [list() for _ in
         range(2)]
recover = [list() for _ in
           range(2)]
positive_line = list()
negative_line = list()

columns_X = ["temp", "dewp", "slp", "wdsp", "prcp", "rh"]
columns_Y = ["increased_confirmed", "increased_deaths", "increased_recovered"]
if query == '1':
    # find relationship
    # calculate pearson's correlation
    for item in columns_Y:
        for var in columns_X:
            ready = select_city.loc[select_city[var].notnull()]
            if statistics.variance(ready[item]) == 0:
                print('No correlation')
            elif statistics.variance(ready[var]) == 0:
                print('No correlation')
            else:
                coef, p = pearsonr(ready[item], ready[var])
                # interpret the significance
                alpha = 0.05
                if p > alpha:
                    print('For ', item, ' and ', var, ' :')
                    print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
                else:
                    print('Samples are correlated (reject H0) p=%.3f' % p)
                    print(coef)
                    if coef < 0:
                        negative_line.append((item, var, coef))
                    else:
                        positive_line.append((item, var, coef))
    negative_line = sorted(negative_line, key=lambda x: x[2])
    positive_line = sorted(positive_line, key=lambda x: -x[2])

    for item in positive_line:
        if item[0] == 'increased_confirmed':
            confirmed[0].append(item)
        elif item[0] == 'increased_deaths':
            death[0].append(item)
        else:
            recover[0].append(item)
    for item in negative_line:
        if item[0] == 'increased_confirmed':
            confirmed[1].append(item)
        elif item[0] == 'increased_deaths':
            death[1].append(item)
        else:
            recover[1].append(item)

    if len(recover[0]) == 0 & len(death[0]) == 0 & len(recover[0]) == 0:
        print('Conclusion: Can not find any correlation for ', city)
    i = 1
    if len(confirmed[0]) != 0:
        plt.figure(i)
        i = i+1
        arr0 = confirmed[0][0][0]
        arr1 = confirmed[0][0][1]
        arr2 = confirmed[1][0][1]
        confirmed_X = select_city[[arr0, arr1, arr2]]
        fig, ((ax1, ax2)) = plt.subplots(1, 2, sharey=True)
        title = 'Increased Confirmed VS '+arr1+' fig1,'+arr2+' fig2'
        fig.suptitle(title, fontsize=20)
        ax1.plot(confirmed_X[arr1], confirmed_X[arr0], 'r*')
        ax2.plot(confirmed_X[arr2], confirmed_X[arr0], 'b*')
        plt.ylabel("Increased Confirmed Cases", fontsize=15)
    if len(death[0]) != 0:
        plt.figure(i)
        i = i+1
        arr0 = death[0][0][0]
        arr1 = death[0][0][1]
        arr2 = death[1][0][1]
        death_X = select_city[[arr0, arr1, arr2]]
        fig, ((ax1, ax2)) = plt.subplots(1, 2, sharey=True)
        title = 'Increased Deaths VS '+arr1+' fig1,'+arr2+' fig2'
        fig.suptitle(title, fontsize=20)
        ax1.plot(death_X[arr1], death_X[arr0], 'r*')
        ax2.plot(death_X[arr2], death_X[arr0], 'b*')
        plt.ylabel("Increased Deaths", fontsize=15)
    if len(recover[0]) != 0:
        plt.figure(i)
        arr0 = recover[0][0][0]
        arr1 = recover[0][0][1]
        arr2 = recover[1][0][1]
        recover_X = select_city[[arr0, arr1, arr2]]
        fig, ((ax1, ax2)) = plt.subplots(1, 2, sharey=True)
        title = 'Increased Recovers VS '+arr1+' fig1,'+arr2+' fig2'
        fig.suptitle(title, fontsize=20)
        ax1.plot(recover_X[arr1], recover_X[arr0], 'r*')
        ax2.plot(recover_X[arr2], recover_X[arr0], 'b*')
        plt.ylabel("Increased Recovers", fontsize=15)
    plt.show()


if query == '2':
    CountryWeather = CityWeather.groupby(['Country', 'Date_x'], as_index=False).agg(
        {'lat': np.mean, 'lon': np.mean, 'temp': np.mean, 'dewp': np.mean, 'slp': np.mean, 'rh': np.mean, 'wdsp': np.mean, 'prcp': np.mean, 'increased_confirmed': np.sum, 'increased_deaths': np.sum, 'increased_recovered': np.sum})

if query == '2':
    Country = set(CountryWeather['Country'])
    Country = sorted(Country)
    print(Country)
    print('You can enter anything names shown above.')
    region = input()
    while region not in Country:
        print('Not Found!.You can enter anything names shown above')
        region = input()
    select_Data = CountryWeather.loc[CountryWeather['Country'] == region]
    wh1 = select_Data[["temp", "dewp", "slp", "wdsp", "prcp", "rh",
                       "increased_confirmed", "increased_deaths", "increased_recovered"]]
    cor = wh1.corr()  # Calculate the correlation of the above variables
    sns.heatmap(cor, square=True)  # Plot the correlation as heat map
    plt.title('Correlation', fontsize=30)

if query == '2':
    # reference from https://www.kaggle.com/concealberti/covid-week1-weather
    columns_X = ["temp", "dewp", "slp", "wdsp", "prcp", "rh"]
    columns_Y = ["increased_confirmed",
                 "increased_deaths", "increased_recovered"]

    weather_PerDay2 = CountryWeather[["temp", "dewp", "slp", "wdsp", "rh",
                                      "prcp", "increased_confirmed", "increased_deaths", "increased_recovered"]]
    weather_PerDay2.replace([np.inf, -np.inf], np.nan, inplace=True)

    weather_PerDay2.data = CovidWeather[columns_X]
    weather_PerDay2.target = CovidWeather[columns_y]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        weather_PerDay2.data, weather_PerDay2.target, random_state=42)

    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train_full.replace([np.inf, -np.inf], np.nan, inplace=True)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(X_train_full)
    X_train_full = imputer.transform(X_train_full)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(X_test)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    rnd_reg = RandomForestRegressor()
    rnd_reg.fit(X_train_scaled, y_train_full)

    predict = rnd_reg.predict(X_test_scaled)
    importances = list(rnd_reg.feature_importances_)
    print('feature importances:', importances)

if query == '2':
    find = max(importances)
    index = importances.index(find)
    explode = [0] * len(importances)
    explode[index] = 0.2
    feature_list = list(weather_PerDay2.data.columns)
    feature_importances = [(feature, importance)
                           for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: -x[1])
    plt.figure(figsize=(20, 10))
    plt.pie(importances, explode=explode, labels=columns_X, shadow=True,
            startangle=90, autopct='%1.1f%%', textprops={'fontsize': 20})
    plt.axis('equal')
    plt.title('The Importance of each Feature', fontsize=30)
    plt.show()

if query == '2':
    feature = feature_importances[0][0]
    plt.scatter(weather_PerDay2[feature],
                weather_PerDay2["increased_confirmed"])
    title = feature+' VS Increased_confirmed'
    plt.title(title, fontsize=25)
    plt.xlabel(feature, fontsize=20)
    plt.ylabel('Increased confirmed cases', fontsize=15)
    plt.show()
    plt.scatter(weather_PerDay2[feature], weather_PerDay2["increased_deaths"])
    title = feature+' VS Increased_deaths'
    plt.title(title, fontsize=25)
    plt.xlabel(feature, fontsize=20)
    plt.ylabel('Increased Death Cases', fontsize=15)
    plt.show()

if query == '2':
    select_Data.index = pd.to_datetime(select_Data['Date_x'])
    date_confirm = select_Data['increased_confirmed'].resample('MS').sum()
    date_confirm = date_confirm.fillna(date_confirm.bfill())

    date_confirm.plot(figsize=(10, 6))
    plt.ylabel('Number of Cases', fontsize=15)
    date_death = select_Data['increased_deaths'].resample('MS').sum()
    date_death = date_death.fillna(date_death.bfill())
    date_death.plot(figsize=(10, 6))
    date_recover = select_Data['increased_recovered'].resample('MS').sum()
    date_recover = date_recover.fillna(date_recover.bfill())
    date_recover.plot(figsize=(10, 6))
    plt.legend(['Confirmed Cases', 'Death Cases', 'Recovered Cases'])
    title = 'Covid19 Situation in '+region
    plt.title(title, fontsize=30)
    plt.grid(True)
    plt.show()

    date_feature = select_Data[feature].resample('MS').mean()
    date_feature = date_feature.fillna(date_feature.bfill())
    date_feature.plot(figsize=(10, 6))
    plt.ylabel(feature, fontsize=15)
    plt.legend([feature])
    title = 'Most Important Features: ' + feature
    plt.title(title, fontsize=30)
    plt.grid(True)
    plt.show()

if query == '2':
    daily = select_Data[['increased_confirmed', 'Date_x']]
    daily = daily.reset_index(drop=True)
    acplot(daily['increased_confirmed'])
    title = 'Autocorrelation of daily confrimed cases( '+region+' )'
    plt.title(title, size=17)
    plt.xlabel('Lag size', size=13)
    plt.ylabel('Autocorrelation', size=13)
    plt.show()
    p, d, q = 11, 2, 0
    date_list = daily.Date_x
    arima = ARIMA(daily['increased_confirmed'], order=(p, d, q))
    arima = arima.fit()
    print(
        f'# ARIMA model fitted\n[Parameters]\np(AR part): {p}, d(I part): {d}, q(MA part): {q}')
    # 2. Print the result report
    # print(arima.summary())

    # 3. Check residual errors
    fig, ax = plt.subplots(figsize=(13, 7))
    plt.plot(daily.Date_x, arima.resid)
    title = 'Residual Errors of ARIMA on Daily Confrimed Cases('+region+' )'
    plt.title(title, size=17)
    plt.xlabel('Date', size=13)
    plt.ylabel('Residual errors', size=13)
    ax.set_xticks(ax.get_xticks()[::int(len(daily.Date_x)/8)])
    plt.show()
    # 1. Check distribution of residual errors
    arima.resid.plot(kind='kde', grid=False)
    plt.title('Residual Errors Distribution', size=17)
    plt.xlabel('Residual Errors', size=13)
    plt.ylabel('Density', size=13)
    plt.show()
    # 2. Check statistics
    print('[Basic statistics]')
    print(arima.resid.describe())

if query == '2':
    # 1. Overlap predictions(+1 step to the last observation) onto the truth
    fig, ax = plt.subplots(figsize=(13, 7))
    plt.plot(daily.Date_x, daily.increased_confirmed,
             color='#33322B', ls=':', lw=3)
    plt.plot(daily.Date_x, arima.predict())
    title = 'ARIMA (one-step forecasting for every date) in ' + region
    plt.title(title, size=17)
    plt.xlabel('Date', size=13)
    plt.ylabel('Number of daily confirmed cases', size=13)
    ax.set_xticks(ax.get_xticks()[::int(len(daily.Date_x)/8)])
    plt.legend(['Truth', 'Prediction'], loc='upper left')
    plt.show()

    # 2. Check scores
    meae = metrics.median_absolute_error(
        daily.increased_confirmed, arima.predict())
    mae = metrics.mean_absolute_error(
        daily.increased_confirmed, arima.predict())
    rmse = metrics.mean_squared_error(
        daily.increased_confirmed, arima.predict())
    rmse = math.sqrt(rmse)
    scores = pd.DataFrame(
        {'rmse': rmse, 'mae': mae, 'meae': meae}, index=['score']
    )
    display(scores)
    print('- RMSE: Root Mean Sqaure Error\
        \n- MAE: Mean Absolute Error\
        \n- MEAE: Median Absolute Error\
          ')

# find the best parameter
# reference from : https://levelup.gitconnected.com/simple-forecasting-with-auto-arima-python-a3f651271965
# evaluate an ARIMA model for a given order (p,d,q)


def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    history = [x for x in X]
    # make predictions
    model = ARIMA(history, order=arima_order)
    model_fit = model.fit()
    start = 0
    end = len(X)-1
    predictions = model_fit.predict(start, end, typ='levels')
    # calculate out of sample error
    error = metrics.median_absolute_error(X, predictions)
    return error
# evaluate combinations of p, d and q values for an ARIMA model


def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                mse = evaluate_arima_model(dataset, order)
                if mse < best_score:
                    best_score, best_cfg = mse, order
                print('ARIMA%s MSE=%.3f' % (order, mse))

    print('Best ARIMA%sMedian absolute error=%.3f' % (best_cfg, best_score))
    return best_cfg


if query == '2':
    # evaluate parameters
    p_values = [2, 5, 11]
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    best_cgf = evaluate_models(
        daily['increased_confirmed'], p_values, d_values, q_values)

if query == '2':
    arima = ARIMA(daily['increased_confirmed'], order=best_cgf)
    arima = arima.fit()

    arima_pred = arima.predict(start=0, end=(
        len(daily)-1), typ='levels').rename('Forecast')

    # 2. Overlap predictions(+1 step to the last observation) onto the truth
    fig, ax = plt.subplots(figsize=(13, 7))

    plt.plot(daily.Date_x, daily.increased_confirmed,
             color='#33322B', ls=':', lw=3)
    plt.plot(daily.Date_x, arima_pred)
    plt.title('ARIMA (with parametrs found by gird search)', size=17)
    plt.xlabel('Date', size=13)
    plt.ylabel('Number of cases', size=13)
    ax.set_xticks(ax.get_xticks()[::int(len(daily.Date_x)/8)])
    plt.legend(['Truth', 'Prediction'], loc='upper left')
    plt.show()
    # Evaluation
    # 2. Check scores
    diff_meae = metrics.median_absolute_error(
        daily.increased_confirmed, arima_pred)
    diff_mae = metrics.mean_absolute_error(
        daily.increased_confirmed, arima_pred)
    diff_rmse = metrics.mean_squared_error(
        daily.increased_confirmed, arima_pred)
    diff_rmse = math.sqrt(diff_rmse)
    scores = pd.DataFrame(
        {'rmse': diff_rmse, 'mae': diff_mae, 'meae': diff_meae}, index=['score']
    )
    print(scores)
    print('- RMSE: Root Mean Sqaure Error\
        \n- MAE: Mean Absolute Error\
        \n- MEAE: Median Absolute Error\
          ')
    # Performance gain checker
    if meae == diff_meae:
        print('11,2,0 is the best parameter.')
    else:
        print('Median Absolute Error decreases ',
              round((meae-diff_meae)/meae, 4)*100, '%')

if query == '2':
    tscv = TimeSeriesSplit(n_splits=5)
    meae = []
    predictions = list()
    for train_index, test_index in tscv.split(daily):
        cv_train, cv_test = daily.iloc[train_index], daily.iloc[test_index]
        # cv_train=cv_train.reset_index(drop=True)
        cv_test = cv_test.reset_index(drop=True)
        model = ARIMA(cv_train['increased_confirmed'], order=best_cgf)
        result = model.fit()
        start = len(cv_train)
        end = len(cv_train)+len(cv_test)-1
        predictions = result.predict(
            start, end, typ='levels').rename("Predictions")
        meae.append(metrics.median_absolute_error(
            predictions, cv_test['increased_confirmed']))
    meae.remove(max(meae))
    meae.remove(min(meae))
    print('Mean median absolute error is ', statistics.mean(meae))

if query == '2':
    # predict
    start = len(daily)
    end = (len(daily)-1)+1*7
    model = ARIMA(daily['increased_confirmed'], order=best_cgf)
    result = model.fit()
    forecast = result.predict(start=start, end=end,
                              typ='levels').rename('Forecast')
    Date = pd.date_range(start="2020-08-11", end="2020-08-17")

    fig, ax = plt.subplots(figsize=(13, 7))
    plt.plot(daily.Date_x, daily.increased_confirmed)
    title = 'Increased Confirmed cases in '+region
    plt.title(title, size=17)
    plt.xlabel('Date', size=13)
    plt.ylabel('Number of cases', size=13)
    ax.set_xticks(ax.get_xticks()[::int(len(daily.Date_x)/8)])
    plt.legend(['True'], loc='upper left')

    fig, ax = plt.subplots(figsize=(13, 7))
    plt.plot(Date, forecast)
    title = 'Forecast confirmed cases for one week in '+region
    plt.title(title, size=17)
    plt.xlabel('Date', size=13)
    plt.ylabel('Number of cases', size=13)

    plt.legend(['Prediction'], loc='upper left')

    plt.show()

if query == '3':
        # Groping the same cities and countries together along with their successive dates.
    covid_cases = covid_cases[['Date_x', 'State',
                               'Country', 'Confirmed', 'Deaths', 'Recovered']]
    country_list = covid_cases['Country'].unique()

    country_grouped_covid = covid_cases[0:1]

    for country in country_list:
        test_data = covid_cases['Country'] == country
        test_data = covid_cases[test_data]
        country_grouped_covid = pd.concat(
            [country_grouped_covid, test_data], axis=0)

    country_grouped_covid.reset_index(drop=True)

    # Replacing NaN Values in Province/State with a string "Not Reported"
    country_grouped_covid['State'].replace(
        np.nan, "Not Reported", inplace=True)

    # Printing the dataset
    # print(country_grouped_covid)

# reference from: https://www.kaggle.com/aestheteaman01/demographics-observation-for-pandemic-escalation
# #Which-populations-are-at-risk-of-contracting-COVID-19?
if query == '3':
    # Creating the interactive map
    py.init_notebook_mode(connected=True)

    # GroupingBy the dataset for the map
    formated_gdf = covid_cases.groupby(['Date_x', 'Country'])[
        'Confirmed', 'Deaths', 'Recovered'].max()
    formated_gdf = formated_gdf.reset_index()
    formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date_x'])
    formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')

    formated_gdf['log_ConfirmedCases'] = np.log(formated_gdf.Confirmed + 1)
    formated_gdf['log_Deaths'] = np.log(formated_gdf.Deaths + 1)

    # Plotting the figure

    fig = px.choropleth(formated_gdf, locations="Country", locationmode='country names',
                        color="log_ConfirmedCases", hover_name="Country", projection="mercator",
                        animation_frame="Date", width=1000, height=800,
                        color_continuous_scale=px.colors.sequential.Viridis,
                        title='The Spread of COVID-19 Cases Across World')

    # Showing the figure
    fig.update(layout_coloraxis_showscale=True)
    py.offline.iplot(fig)

if query == '3':
    # reference from: https://www.kaggle.com/aestheteaman01/demographics-observation-for-pandemic-escalation
    # #Which-populations-are-at-risk-of-contracting-COVID-19?
    fig = px.choropleth(formated_gdf, locations="Country", locationmode='country names',
                        color="log_Deaths", hover_name="Country", projection="mercator",
                        animation_frame="Date", width=1000, height=800,
                        color_continuous_scale=px.colors.sequential.Viridis,
                        title='The COVID-19 Deaths Cases Across World')

    # Showing the figure
    fig.update(layout_coloraxis_showscale=True)
    py.offline.iplot(fig)

# reference from: https://www.kaggle.com/aestheteaman01/demographics-observation-for-pandemic-escalation
# #Which-populations-are-at-risk-of-contracting-COVID-19?
if query == '3':
    country_data_detailed = CityWeather
    country_data_detailed = country_data_detailed.drop(['day_from_jan_first', 'stn', 'STN', 'usaf', 'wban_right', 'country', 'ah', 'month', 'day',
                                                        'wban_left', 'year', 'mo', 'da', 'Date_y', 'Date2'], axis=1)
    country_data_detailed['State'].replace(
        np.nan, "Not Reported", inplace=True)
    country_data_detailed.replace(np.nan, 0, inplace=True)
    temperature_data = country_data_detailed
    # print(country_data_detailed.isnull().sum(axis=0))
    # Checking the dependence of Temperature on Confirmed COVID-19 Cases
    unique_temp = temperature_data['temp'].unique()
    confirmed_cases = []
    deaths = []

    for temp in unique_temp:
        temp_wise = temperature_data['temp'] == temp
        test_data = temperature_data[temp_wise]

        confirmed_cases.append(test_data['Confirmed'].sum())
        deaths.append(test_data['Deaths'].sum())

    # Converting the lists to a pandas dataframe.

    temperature_dataset = {'temp': unique_temp,
                           'Confirmed': confirmed_cases, 'Deaths': deaths}
    temperature_dataset = pd.DataFrame(temperature_dataset)
    # Plotting a scatter plot for cases vs. Temperature

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scattergl(x=temperature_dataset['temp'], y=temperature_dataset['Confirmed'], mode='markers',
                               marker=dict(color=np.random.randn(10000), colorscale='Viridis', line_width=1)), secondary_y=False)

    fig.add_trace(go.Box(x=temperature_dataset['temp']), secondary_y=True)

    fig.update_layout(title='Daily Confirmed Cases (COVID-19) vs. Temperature (Fahrenheit) : Global Figures - January 22 - August 01 2020',
                      yaxis=dict(title='Reported Numbers'), xaxis=dict(title='Temperature in Fahrenheit'))
    fig.update_yaxes(title_text="BoxPlot Range ", secondary_y=True)
    fig.show()

# reference from: https://www.kaggle.com/aestheteaman01/demographics-observation-for-pandemic-escalation
# #Which-populations-are-at-risk-of-contracting-COVID-19?
if query == '3':
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scattergl(x=temperature_dataset['temp'], y=temperature_dataset['Deaths'], mode='markers',
                               marker=dict(color=np.random.randn(10000), colorscale='Viridis', line_width=1)), secondary_y=False)

    fig.add_trace(go.Box(x=temperature_dataset['temp']), secondary_y=True)

    fig.update_layout(title='Daily Deaths Cases (COVID-19) vs. Temperature (Fahrenheit) : Global Figures - January 22 - August 01 2020',
                      yaxis=dict(title='Reported Numbers'), xaxis=dict(title='Temperature in Fahrenheit'))
    fig.update_yaxes(title_text="BoxPlot Range ", secondary_y=True)
    fig.show()
    # building hypothesis test
    # null hypothesis: the temperature affect on COVID-19 remains same over the population data
    sample_size = int(len(temperature_dataset['temp'])*0.1)
    if sample_size > 1000:
        sample_size = 1000
    sample = temperature_dataset['temp'].sample(n=sample_size)
    test = temperature_dataset['temp']
    from scipy.stats import ttest_ind
    stat, p = ttest_ind(sample, test)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Since we get p value > 0.05 we can safely accept our null hypothesis and can conclude, that temperature affect on COVID-19 remains same over the population data.')
    else:
        print('Reject null hypothesis')
