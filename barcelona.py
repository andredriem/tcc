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

barcelona_dataset = pd.read_csv('barcelona-data-sets/accidents_2017.csv')
barcelona_dataset_2 = pd.read_csv('barcelona-data-sets/accidents_2017.csv')

d = dict((v,k) for k,v in enumerate(calendar.month_name))
barcelona_dataset['Month'] = barcelona_dataset['Month'].apply(lambda x: d[x])
barcelona_dataset_2['Month'] = barcelona_dataset_2['Month'].apply(lambda x: d[x])

for label in ['Weekday','District Name', 'Neighborhood Name'
       ]:
    le = preprocessing.LabelEncoder()
    le.fit(barcelona_dataset[label])
    barcelona_dataset[label] = le.transform(barcelona_dataset[label])
    barcelona_dataset_2[label] = barcelona_dataset[label]

# 'District Name','Neighborhood Name'

barcelona_dataset = barcelona_dataset.groupby(['Month','Day'])['Victims'].apply(list).reset_index()
barcelona_dataset_2 = barcelona_dataset_2.groupby(['Month','Day'])['Weekday'].apply(lambda x: list(x)[0]).reset_index()

for i, row in barcelona_dataset.iterrows():
    barcelona_dataset.at[i, 'Victims'] = len(barcelona_dataset.at[i, 'Victims'])




#barcelona_dataset = barcelona_dataset[barcelona_dataset['District Name'] == 0]

#print(barcelona_dataset.to_string())


weather = pd.read_csv('barcelona-data-sets/weather.csv')
weather =  weather.drop(['Y'], axis=1)

from sklearn.preprocessing import MinMaxScaler

for column in weather.columns:
    if column not in ['Y','M','D']:
        scaler = MinMaxScaler()
        weather[column] = scaler.fit_transform(weather[column].values.reshape(-1,1))

barcelona_dataset.sort_values(['Month', 'Day'], inplace=True)

print(barcelona_dataset_2[['Month','Day','Weekday']])
barcelona_dataset = pd.merge(barcelona_dataset, weather,  how='left', left_on=['Month','Day'], right_on = ['M','D'])
barcelona_dataset = pd.merge(barcelona_dataset, barcelona_dataset_2[['Month','Day','Weekday']],  how='left', left_on=['Month','Day'], right_on = ['Month','Day'])
print(barcelona_dataset)

scaler = MinMaxScaler()
barcelona_dataset['Victims'] = scaler.fit_transform(barcelona_dataset['Victims'].values.reshape(-1,1))
scaler = MinMaxScaler()
barcelona_dataset['Day'] = scaler.fit_transform(barcelona_dataset['Day'].values.reshape(-1,1))
scaler = MinMaxScaler()
barcelona_dataset['Month'] = scaler.fit_transform(barcelona_dataset['Month'].values.reshape(-1,1))

scaler = MinMaxScaler()
barcelona_dataset['Weekday'] = scaler.fit_transform(barcelona_dataset['Weekday'].values.reshape(-1,1))


#print(barcelona_dataset.to_string())

"""for label in ['District Name'
       ]:
    le = preprocessing.LabelEncoder()
    le.fit(barcelona_dataset[label])
    barcelona_dataset[label] = le.transform(barcelona_dataset[label])
"""
for label in barcelona_dataset.columns:
    barcelona_dataset[label] = barcelona_dataset[label].astype(float)


"""
min_max_scaler = preprocessing.MinMaxScaler()


barcelona_dataset =pd.DataFrame(
    min_max_scaler.fit_transform(barcelona_dataset),
    columns=barcelona_dataset.columns)
"""
train=barcelona_dataset.head(int(len(barcelona_dataset)*(80/100)))
train_labels = train['Victims']
train_input = train.drop('Victims', axis=1)

test=barcelona_dataset.drop(train.index)
test_labels = test['Victims']
test_input = test.drop('Victims', axis=1)

n_input = 7
b_size = 32
n_features = train_input.shape[1]


generator_test = TimeseriesGenerator(train_input.values, train_labels.values, length=n_input, batch_size=b_size)
generator = TimeseriesGenerator(test_input.values, test_labels.values, length=n_input, batch_size=b_size)
print(generator)
"""
for i in range(len(generator)):
	x, y = generator[i]
	print('%s => %s' % (x, y))
"""

model = Sequential()
model.add(layers.LSTM(256, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(16))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit_generator(generator, steps_per_epoch=None, epochs=200 )


accuracy_list = []
for series in  TimeseriesGenerator(test_input.values, test_labels.values, length=n_input, batch_size=1):
    prediction = [model.predict(series[0], verbose=1)[0][0], series[1][0]]
    accuracy_list.append(prediction)
    print(prediction)

import matplotlib.pyplot as plt
plt.plot([x[1] for x in accuracy_list], color='red')
plt.plot([x[0] for x in accuracy_list])
plt.ylabel('Number of accidents')
plt.show()

print(prediction)

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
