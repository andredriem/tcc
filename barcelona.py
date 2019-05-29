import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from tensorflow import keras


barcelona_dataset = pd.read_csv('barcelona-data-sets/accidents_2017.csv')
barcelona_dataset = barcelona_dataset.drop(['Day', 'Longitude', 'Latitude', 'Street', "Id", 
    'Vehicles involved',  'Mild injuries', 'Serious injuries'], axis=1)
print (barcelona_dataset.columns)

for label in ['District Name', 'Neighborhood Name', 'Weekday', 'Month',
       'Part of the day']:
    le = preprocessing.LabelEncoder()
    le.fit(barcelona_dataset[label])
    barcelona_dataset[label] = le.transform(barcelona_dataset[label])

for label in barcelona_dataset.columns:
    barcelona_dataset[label] = barcelona_dataset[label].astype(float)



"""
min_max_scaler = preprocessing.MinMaxScaler()


barcelona_dataset =pd.DataFrame(
    min_max_scaler.fit_transform(barcelona_dataset),
    columns=barcelona_dataset.columns)
"""
train = barcelona_dataset.sample(frac=0.8)
train_labels = train['Victims']
train_input = train.drop('Victims', axis=1)

test = barcelona_dataset.drop(train.index)
test_labels = test['Victims']
test_input = test.drop('Victims', axis=1)


model = keras.Sequential([
    keras.layers.Dense(len(train.columns)),
    keras.layers.Dense(258, activation=tf.nn.relu),
    keras.layers.Dense(258, activation=tf.nn.relu),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', 
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(train_input.values, train_labels.values, epochs=20)

test_loss, test_acc = model.evaluate(test_input, test_labels)

print(model.predict(train_input.values))



print('Test accuracy:', test_acc)
