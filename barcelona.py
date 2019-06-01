import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


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


barcelona_dataset = barcelona_dataset[barcelona_dataset['District Name'] == 7]
"""
min_max_scaler = preprocessing.MinMaxScaler()


barcelona_dataset =pd.DataFrame(
    min_max_scaler.fit_transform(barcelona_dataset),
    columns=barcelona_dataset.columns)
"""
train_labels = barcelona_dataset['Victims']
train_input = barcelona_dataset.drop('Victims', axis=1)

n_input = 365
n_features = train_input.shape[1]
generator = TimeseriesGenerator(train_input.values, train_labels.values, length=n_input, batch_size=1)
print(generator)

"""
for i in range(len(generator)):
	x, y = generator[i]
	print('%s => %s' % (x, y))
"""

model = Sequential()
model.add(LSTM(1000, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
# fit model
model.fit_generator(generator, steps_per_epoch=1, epochs=500)

"""

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
"""
