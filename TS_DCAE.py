from keras.models import Model 
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv1D, UpSampling1D
import np_utils
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
import os
from sklearn.preprocessing import StandardScaler
import os
import datetime
from sklearn.model_selection import train_test_split

poor = pd.read_csv("C:/Users/met48/Desktop/TS-Clustering/SimData/bank_reserves_outputs_poor.csv", header=None)
middle = pd.read_csv("C:/Users/met48/Desktop/TS-Clustering/SimData/bank_reserves_outputs_middle.csv", header=None)
rich = pd.read_csv("C:/Users/met48/Desktop/TS-Clustering/SimData/bank_reserves_outputs_rich.csv", header=None)
sc = StandardScaler()
bank_reserves = []
for i in np.arange(0, poor.shape[0]):
    sample = pd.concat([poor.iloc[i], middle.iloc[i]], axis=0).T
    sample = pd.concat([sample, rich.iloc[i]], axis=0).T
    sample_std = sc.fit_transform(sample.to_frame())
    bank_reserves.append(sample_std)
	
bank_reserves_train, bank_reserves_test = train_test_split(bank_reserves, test_size=0.20, random_state=42)

input = Input(shape=(303,1))

encoded = Conv1D(filters=10, kernel_size=32, activation='relu', padding='same')(input)
encoded = Conv1D(filters=10, kernel_size=64, activation='relu', padding='same')(encoded)
encoded = Conv1D(filters=10, kernel_size=128, activation='relu', padding='same')(encoded)
encoded = Conv1D(filters=10, kernel_size=128, activation='relu', padding='same')(encoded)

encoded = Dense(384, activation='relu')(encoded)
decoded = Dense(5, activation='relu')(encoded)
decoded = Dense(384, activation='relu')(decoded)

decoded = UpSampling1D(size=2)(decoded)
decoded = Conv1D(filters=10, kernel_size=128, activation='relu')(decoded)
decoded = UpSampling1D(size=2)(decoded)
decoded = Conv1D(filters=10, kernel_size=128, activation='relu')(decoded)
decoded = UpSampling1D(size=2)(decoded)
decoded = Conv1D(filters=10, kernel_size=64, activation='relu')(decoded)
decoded = UpSampling1D(size=2)(decoded)
decoded = Conv1D(filters=10, kernel_size=32, activation='relu')(decoded)

autoencoder = Model(input, decoded)
autoencoder.summary()

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

autoencoder.fit(bank_reserves_train, bank_reserves_train,
 epochs=20,
 batch_size=8,
 shuffle=True,
 validation_data=(bank_reserves_test, bank_reserves_test),
 verbose=1)
 
#encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=4).output)
#encoded_ts = encoder.predict(bank_reserves_test)

#encoded_input = Input(shape=(5,))  # 5 is the size of your latent space
#decoded_layer = autoencoder.layers[5](encoded_input)
#for layer in autoencoder.layers[6:]:
#    decoded_layer = layer(decoded_layer)

#decoder = Model(inputs=encoded_input, outputs=decoded_layer)
#decoded_ts = decoder.predict(encoded_ts)