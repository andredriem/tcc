import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from scipy.stats import kendalltau

import calendar
import concise as c

barcelona_dataset = pd.read_csv('2011.csv')
barcelona_dataset['Year'] = 2011

temp = pd.read_csv('2012.csv')
temp['Year'] = 2012
barcelona_dataset = pd.concat([barcelona_dataset, temp])

temp = pd.read_csv('2013.csv')
temp['Year'] = 2013
barcelona_dataset = pd.concat([barcelona_dataset, temp])

temp = pd.read_csv('2014.csv')
temp['Year'] = 2014
barcelona_dataset = pd.concat([barcelona_dataset, temp])

temp = pd.read_csv('2015.csv')
temp['Year'] = 2015
barcelona_dataset = pd.concat([barcelona_dataset, temp])

temp = pd.read_csv('2016.csv')
temp['Year'] = 2016
barcelona_dataset = pd.concat([barcelona_dataset, temp])

temp = pd.read_csv('2017.csv')
temp['Year'] = 2017
barcelona_dataset = pd.concat([barcelona_dataset, temp])

temp = pd.read_csv('2018.csv')
temp['Year'] = 2018
barcelona_dataset = pd.concat([barcelona_dataset, temp])

barcelona_dataset.to_csv('full.csv')