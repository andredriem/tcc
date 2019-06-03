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
barcelona_dataset['Year'] = 2017
barcelona_2018 = pd.read_csv('barcelona-data-sets/2018_accidents.csv')
barcelona_2018['Year'] = 2018

barcelona_dataset = pd.concat([barcelona_dataset,barcelona_2018])

print("TAIL:")
print(barcelona_dataset.tail(1))

def month(x):
    try:
        return d[x]
    except:
        return x





d = dict((v,k) for k,v in enumerate(calendar.month_name))
barcelona_dataset['Month'] = barcelona_dataset['Month'].apply(month)
print("TAIL:")
print(barcelona_dataset.tail(1))
barcelona_dataset_2 = barcelona_dataset.copy()

for label in ['Weekday']:
    le = preprocessing.LabelEncoder()
    le.fit(barcelona_dataset[label])
    barcelona_dataset[label] = le.transform(barcelona_dataset[label])
    barcelona_dataset_2[label] = barcelona_dataset[label]

# 'District Name','Neighborhood Name'

print("TAIL PRE GROUPBY:")
print(barcelona_dataset.groupby(['Year','Month','Day'], sort=False)['Victims'].apply(list))
barcelona_dataset = barcelona_dataset.groupby(['Year','Month','Day'], sort=False)['Victims'].apply(list).reset_index()
print("TAIL WITH 2018:")
print(barcelona_dataset.tail())
barcelona_dataset_2 = barcelona_dataset_2.groupby(['Year','Month','Day'])['Weekday'].apply(lambda x: list(x)[0]).reset_index()

for i, row in barcelona_dataset.iterrows():
    barcelona_dataset.at[i, 'Victims'] = len(barcelona_dataset.at[i, 'Victims'])




#barcelona_dataset = barcelona_dataset[barcelona_dataset['District Name'] == 0]

#print(barcelona_dataset.to_string())


weather = pd.read_csv('barcelona-data-sets/weather_full.csv')
holidays = pd.read_csv('barcelona-data-sets/holidays.csv')
from sklearn.preprocessing import MinMaxScaler

for column in weather.columns:
    if column not in ['Y','M','D']:
        scaler = MinMaxScaler()
        weather[column] = scaler.fit_transform(weather[column].values.reshape(-1,1))

barcelona_dataset.sort_values(['Year','Month', 'Day'], inplace=True)
print("FINAL TAIL WITH 2018", barcelona_dataset.tail())

print(barcelona_dataset_2[['Month','Day','Weekday']])
barcelona_dataset = pd.merge(barcelona_dataset, weather,  how='left', left_on=['Year','Month','Day'], right_on = ['Y','M','D'])
barcelona_dataset = pd.merge(barcelona_dataset, barcelona_dataset_2[['Year','Month','Day','Weekday']],  how='left', left_on=['Year','Month','Day'], right_on = ['Year','Month','Day'])
barcelona_dataset= barcelona_dataset.drop(['Y','M','D'], axis=1)
barcelona_dataset = pd.merge(barcelona_dataset, holidays,  how='left', left_on=['Year','Month','Day'], right_on = ['Y','M','D'])
barcelona_dataset= barcelona_dataset.drop(['Y','M','D'], axis=1)
barcelona_dataset.to_csv('barcelona-data-sets/refined.csv')


#barcelona_dataset.drop(barcelona_dataset.tail(60).index,inplace=True)

print(barcelona_dataset)

victim_scaler = MinMaxScaler()
first = barcelona_dataset['Victims'][0]
barcelona_dataset['Victims'] = victim_scaler.fit_transform(barcelona_dataset['Victims'].values.reshape(-1,1))
second = barcelona_dataset['Victims'][0]
scale_coeficient = first/second 
scaler = MinMaxScaler()
barcelona_dataset['Day'] = scaler.fit_transform(barcelona_dataset['Day'].values.reshape(-1,1))
scaler = MinMaxScaler()
barcelona_dataset['Month'] = scaler.fit_transform(barcelona_dataset['Month'].values.reshape(-1,1))

scaler = MinMaxScaler()
barcelona_dataset['Weekday'] = scaler.fit_transform(barcelona_dataset['Weekday'].values.reshape(-1,1))

scaler = MinMaxScaler()
barcelona_dataset['Year'] = scaler.fit_transform(barcelona_dataset['Year'].values.reshape(-1,1))

scaler = MinMaxScaler()
barcelona_dataset['Yesterday accidents'] = scaler.fit_transform(barcelona_dataset['Yesterday accidents'].values.reshape(-1,1))


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

train=barcelona_dataset.head(int(len(barcelona_dataset)*(70/100)))
train_labels = train['Victims']
train_input = train.drop('Victims', axis=1)

test=barcelona_dataset.drop(train.index)
test_labels = test['Victims']
test_input = test.drop('Victims', axis=1)

n_input = 10
b_size = 32
n_features = train_input.shape[1]


generator = TimeseriesGenerator(train_input.values, train_labels.values, length=n_input, batch_size=b_size, shuffle=True)
generator_test = TimeseriesGenerator(test_input.values, test_labels.values, length=n_input, batch_size=b_size)
print("generator_len")
print(len(TimeseriesGenerator(train_input.values, train_labels.values, length=n_input, batch_size=b_size)))
"""
for i in range(len(generator)):
	x, y = generator[i]
	print('%s => %s' % (x, y))
"""

model = Sequential()
model.add(layers.LSTM(16, activation='sigmoid', input_shape=(n_input, n_features)))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(optimizer='adam', loss='mae')
# fit model
model.fit_generator(generator, epochs=200, use_multiprocessing=True )


accuracy_list = []
predicted = []
expected = []
i = 0
for series in  TimeseriesGenerator(test_input.values, test_labels.values, length=n_input, batch_size=1):
    if i== 30:
        break
    prediction = [model.predict(series[0], verbose=1)[0][0], series[1][0]]

    predicted.append(prediction[0])
    expected.append(prediction[1])
    accuracy_list.append(prediction)
    print(prediction)
    i += 1

import matplotlib.pyplot as plt
plt.plot([x[1]*scale_coeficient for x in accuracy_list], color='red')
plt.plot([x[0]*scale_coeficient for x in accuracy_list])
plt.ylabel('Number of accidents')
plt.show()

print("Mean Squared error all list")
print(sum([(p[0] - p[1])**2 for p in accuracy_list])/len(accuracy_list))
print("Mean Absolute error all list")
print(sum([abs(p[0] - p[1]) for p in accuracy_list])/len(accuracy_list))
print("Mean Squared error when input has full size(%(n_input)s)"%locals() )
print(sum([(p[0] - p[1])**2 for p in accuracy_list[n_input:]])/len(accuracy_list[n_input:]))

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
