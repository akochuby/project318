#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 01:45:02 2017

@author: akochuby
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack, ndimage
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, Normalizer
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

import time

start_time = time.time()

VAR_NUM = -27

directory_w = '/home/akochuby/Project318/w/'
directory_i = '/home/akochuby/Project318/i/'

def load_weather_data():

    # Read all the weather files and concatenate them into a single file

    files = os.listdir(directory_w)
    rows_to_skip = np.linspace(0, 15, 16)
    df_list = [pd.read_csv(directory_w + file, skiprows=rows_to_skip) for file in files]
    weather = pd.concat(df_list)
    weather = weather.reset_index(drop=True)

    # Remove columns that are always null or are repetitions of datetime
    # This reduces the size of the data frame almost by 2 times
    weather = weather.drop(['Temp Flag', 'Visibility Flag',
                            'Dew Point Temp Flag', 'Wind Spd Flag',
                            'Rel Hum Flag', 'Stn Press Flag',
                            'Wind Dir Flag', 'Hmdx Flag', 'Wind Chill Flag',
                            'Year', 'Month', 'Day', 'Time'], axis=1)

    # Remove rows where Data Quality is missing(found 3 with all values NaN)
    # After, remove Data Quality colums as it is identical for all other rows
    weather = weather[~weather['Data Quality'].isnull()].drop('Data Quality', axis=1)

    # Convert Date/Time to datetime and rename the remaining columns
    weather['Date/Time'] = weather['Date/Time'].apply(pd.to_datetime)
    weather.columns = ['datetime', 'temp', 'dew_temp',
                       'rel_hum', 'wind_dir', 'wind_spd',
                       'visibility', 'pressure', 'hmdx', 'wind_chill', 'weather']

    return weather


def load_image_data():

    # Get all the image names from directory
    files = os.listdir(directory_i)

    # Perform a 2D FFT on the images and find most variant frequencies(VAR_NUM)
    # Those will be used as input for data frame
    image = ndimage.imread(directory_i + files[0], mode='L')
    fourier = np.zeros((image.shape[0], image.shape[1], len(files)))
    images = np.zeros((len(files), image.shape[0], image.shape[1]))
    for i in range(0, len(files)):
        image = ndimage.imread(directory_i + files[i], mode='L')
        fourier[:, :, i] = abs(fftpack.fft2(image))
        images[i, :, :] = image

    variance = np.var(fourier, axis=2)
    most_variant_indexes = np.argpartition(variance.flatten(), VAR_NUM)[VAR_NUM:]
    selected_cells = np.vstack(np.unravel_index(most_variant_indexes, variance.shape)).T

    # Create a data frame with selected frequencies for each image
    # Additionally, add filename and datetime columns
    image_data_furier = [vectorize(fourier[:, :, i], selected_cells) for i in range(0, len(files))]
    image_data_furier = pd.DataFrame.from_records(image_data_furier)
    image_data_furier['filename'] = files
    image_data_furier['datetime'] = image_data_furier['filename'].apply(name_to_date)

    return image_data_furier, images


# Converts image name to date
def name_to_date(d):
    return pd.to_datetime(d[7:21])


def date_to_name(n):
    return 'katkam-' + n.strftime('%Y%m%d%h%M%s') + '.jpg'


def clear_weather(w):
    a = w.split(',')
    for i in range(0, len(a)):
        word = a[i]
        if 'Clear' in word:
            a[i] = 'Clear'
        elif 'Cloudy' in word:
            a[i] = 'Cloudy'
        elif 'Rain' in word:
            a[i] = 'Rain'
        elif 'Fog' in word:
            a[i] = 'Fog'
        elif 'Snow' in word:
            a[i] = 'Snow'
        elif 'Hail' in word:
            a[i] = 'Hail'
        else:
            a[i] = word
    return a


# Turns an image into a feature vector based on
# the most variant frequencies
def vectorize(matrix, selected_cells):
    return np.array([matrix[cell[0], cell[1]] for cell in selected_cells], dtype=float)


def transform_weather(data):
    data = data[~data['weather'].isnull()]
    data['dew_temp_diff'] = data['temp'] - data['dew_temp']
    data['weather'] = data['weather'].apply(clear_weather)
    return data


def fourier(image):
    return abs(fftpack.fft2(image))


def main():

    image_data_fourier, images = load_image_data()
    weather_data = load_weather_data()
    weather_data = transform_weather(weather_data)

    model = make_pipeline(
        StandardScaler(),     # Around 64%
        #MinMaxScaler(),        # Around 63%
        #Normalizer(),           # Around 65%
        PCA(120),
        OneVsRestClassifier(KNeighborsClassifier(n_neighbors=7))
        #MLPClassifier(50)
        #KNeighborsClassifier(n_neighbors=5)  #7  100 == 68    5 120 == 68.5    9 340 == 69  13 300 == 68.8
        #SVC(kernel='linear', C=1)  # Does not support multilabel, also runs endlessly
    )

    join = image_data_fourier[['datetime', 'filename']]
    weather_data = pd.merge(weather_data, join, on='datetime')

    train_weather = weather_data[~weather_data['weather'].isnull()]

    files = train_weather['filename']

    train_image = [ndimage.imread(directory_i + file, mode='L') for file in files]
    train_image = np.array(train_image)
    train_image = np.reshape(train_image, [train_image.shape[0], train_image.shape[1] * train_image.shape[2]])



    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(train_weather['weather'])

    res = mlb.classes_

    X_train, X_test, y_train, y_test = train_test_split(train_image, y)
    model.fit(X_train, y_train)



    print(classification_report(y_test, model.predict(X_test), target_names=res))

    tmp = pd.DataFrame()

    tmp['real'] = mlb.inverse_transform(y_test)
    tmp['fake'] = mlb.inverse_transform(model.predict(X_test))

    print(accuracy_score(y_test, model.predict(X_test)))
    tmp.to_csv('image_data.csv')
    #weather_data = weather_data.groupby('weather').mean()

    #images.to_csv('weather_data.csv')
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()




