import torch
import torchvision
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from PIL import Image


def create_model(my_learning_rate):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(units=16, activation='relu'))
	model.add(tf.keras.layers.Dropout(rate=0.1))
	model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
	model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
								loss="sparse_categorical_crossentropy",
								metrics=['accuracy'])
	return model        


def train_model(model, train_features, train_label, epochs,
								batch_size=None, validation_split=0.1):
	history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
											epochs=epochs, shuffle=True, 
											validation_split=validation_split)
	epochs = history.epoch
	hist = pd.DataFrame(history.history)

	return epochs, hist


def plot_curve(epochs, hist, list_of_metrics):
	plt.figure()
	plt.xlabel("Epoch")
	plt.ylabel("Value")

	for m in list_of_metrics:
		x = hist[m]
		plt.plot(epochs[1:], x[1:], label=m)

	plt.legend()
	plt.show()


np.set_printoptions(linewidth = 200)
train = pd.read_csv('d1.csv')
test = pd.read_csv('mnist_test.csv')

X = train.iloc[:,1:].values / 255.0
X_test = test.iloc[:,1:].values / 255.0
y = train['label'].values
y_test = test['label'].values


learning_rate = 0.003
epochs = 50
batch_size = 4000
validation_split = 0.2

model = create_model(learning_rate)

epochs, hist = train_model(model, X, y,  epochs, batch_size, validation_split)

list_of_metrics_to_plot = ['accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)

model.evaluate(x=X_test, y=y_test, batch_size=batch_size)
model.save('v1')


loaded_model = tf.keras.models.load_model('v1')

loaded_model.predict(<element>)


