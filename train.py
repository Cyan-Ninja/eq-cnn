"""
Predict the P-wave first motion polarities of microseismic events
with single trace information using CNN.

https://wiki.seg.org/wiki/Phase_and_polarity_assessment_of_seismic_data#Types_of_Polarity

American polarity: positive polarity (impedance) is linked to a peak (positive amplitude)

European polarity: opposite of the American one, which means a positive polarity (impedance) associated with a trough (negative amplitude)
"""

import numpy as np
import matplotlib.pyplot as plt

# dataset reading
#from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
import tables
import pandas as pd

# plotting and tests
#import scipy
#import obsplus
#import obspy
#from typing import Union, Optional, List, Mapping, Dict, Any
#from vector import vector, plot_peaks
#import pytest
# from scipy.signal import find_peaks,peak_prominences
# from obsplus.utils.time import to_utc

# tensorflow
import tensorflow as tf
import keras
from keras import regularizers
from keras import optimizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, GlobalMaxPooling1D, Flatten
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
import operator

# time and os
import scipy.io as sio
np.random.seed(7)
import os
import time

time_start = time.time()

#os.environ['CUDA_VISIBLE_DEVICES']='2' # CUDA GPUs
#tf.executing_eagerly() # eager execution (dev-only)

def read_data(path, name) -> np.array:
	with tables.open_file(dataset_path) as store:
		out = store.root[name]['data'][:]
		# print datasets print(store.root)
	return out

def read_labels(path, name) -> pd.DataFrame:
	"""Read the labels of the dataset"""
	df = pd.read_hdf(path, f"{name}/index")
	return df

if __name__ == "__main__":
	# dataset variables
	dataset_path = 'dataset.h5'
	dataset_name = 'R2'

	# read data and labels
	data = read_data(dataset_path, dataset_name)
	df = read_labels(dataset_path, dataset_name)

	# # slice data/labels and plot
	# num_traces_to_plot = 5
	#
	# sub_data = data[:num_traces_to_plot] # data
	# df_sub = df.iloc[:num_traces_to_plot] # labels
	#
	# sub_z = data[:, :, 0] # just grab z component sub_data[:, :, 0]
	#
	# # example on data
	# print('Data: {}'.format(data[0]))
	# print('Time: {}'.format(df['time'][0]))
	# print('Polarity: {}'.format(df['polarity'][0]))
	# print('Z-Only: {}'.format(sub_z[0]))

	# # add sample rate
	# def _get_pick_samples(df):
	#	 """get the pick sample relative to start of trace"""
	#	 dt = (df['time'] - df['starttime']).dt.total_seconds()
	#	 out = dt * df['sampling_rate']
	#	 return out
	#
	# df['pick_sample'] =_get_pick_samples(df)

	# get result from model

	# organise data for model

	# batch variables
	train_test_ratio = 0.75
	batch_size = 8
	num_classes = 1 # [neg, amb, pos]
	epochs = 200
	# image dimensions
	img_rows, img_cols = 1, 6501
	input_shape = img_cols

	# initial data
	x_data = data[:, :, 0] # data (z channel only)
	y_data = np.copy(df['polarity']) # labels (unlinked array)

	# convert polarity data to an 1/0 integer instead of string
	for i in range(len(y_data)):
		if y_data[i] == 'positive':
			y_data[i] = 1
			#y_data[i] = 2
		elif y_data[i] == 'negative':
			y_data[i] = 0
			#y_data[i] = 0
		else: # '' is ambiguous
			y_data[i] = 0.5
			#y_data[i] = 1

	# convert numpy arrays to tensor
	x_data = tf.convert_to_tensor(x_data.astype('float32'))
	y_data = tf.convert_to_tensor(y_data.astype('float32'))
	# shuffle
	indices = tf.range(start=0, limit=tf.shape(x_data)[0], dtype=tf.int32)
	idx = tf.random.shuffle(indices)
	x_data = tf.gather(x_data, idx)
	y_data = tf.gather(y_data, idx)
	# split
	even_split = int(np.floor(len(x_data) / 2) * 2)
	x_train, x_test = tf.split(x_data[:even_split], num_or_size_splits=2, axis=0)
	y_train, y_test = tf.split(y_data[:even_split], num_or_size_splits=2, axis=0)

	# the data, shuffled and split between train and test sets
	print('x_train- Shape:', x_train.shape, 'Sample:', x_train[0])
	print('y_train- Shape:', y_train.shape, 'Sample:', y_train[0])

	# convert class vectors to binary class matrices as [[b, b, b], ... [b, b, b]] - kept instead as 0/0.5/1 for neg/amb/pos as [f, ... f]
	#y_train = keras.utils.to_categorical(y_train, num_classes)
	#y_test = keras.utils.to_categorical(y_test, num_classes)

	# loss history class
	class LossHistory(keras.callbacks.Callback):
		def on_train_begin(self, logs={}):
			self.losses = {'batch':[], 'epoch':[]}
			self.accuracy = {'batch':[], 'epoch':[]}
			self.val_loss = {'batch':[], 'epoch':[]}
			self.val_acc = {'batch':[], 'epoch':[]}

			def on_batch_end(self, batch, logs={}):
				self.losses['batch'].append(logs.get('loss'))
				self.accuracy['batch'].append(logs.get('acc'))
				self.val_loss['batch'].append(logs.get('val_loss'))
				self.val_acc['batch'].append(logs.get('val_acc'))

				def on_epoch_end(self, batch, logs={}):
					self.losses['epoch'].append(logs.get('loss'))
					self.accuracy['epoch'].append(logs.get('acc'))
					self.val_loss['epoch'].append(logs.get('val_loss'))
					self.val_acc['epoch'].append(logs.get('val_acc'))

					def loss_plot(self, loss_type):
						iters = range(len(self.losses[loss_type]))
						plt.figure()
						# acc
						plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
						# loss
						plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
						if loss_type == 'epoch':
							# val_acc
							plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
							# val_loss
							plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
							plt.grid(True)
							plt.xlabel(loss_type)
							plt.ylabel('acc-loss')
							plt.legend(loc="upper right")
							plt.show()

	# class Split(keras.layers.Layer):
	# 	def __init__(self, input_dim=3):
	# 		super(Split, self).__init__()
	#
	# 	def call(self, inputs):
	# 		result = inputs[0,:,:]
	# 		result = tf.expand_dims(result, axis=0, name=None)
	# 		#result = tf.squeeze(result, axis=None, name=None)
	# 		return result

	# model
	model = Sequential()
	model.add(Conv1D(32, kernel_size=11,
		activation='relu',
		input_shape=(1, input_shape, 1)))
	model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
	# model.add(Split())
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(0.25))

	model.add(Conv1D(64, kernel_size=9,
		activation='relu',
		input_shape=(1, input_shape, 1)))
	model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
	# model.add(Split())
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(0.25))

	model.add(Conv1D(128, kernel_size=5,
		activation='relu',
		input_shape=(1, input_shape, 1)))
	model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
	# model.add(Split())
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(0.25))

	model.add(Conv1D(256, kernel_size=3,
		activation='relu',
		input_shape=(1, input_shape, 1)))
	model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
	# model.add(Split())
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(0.25))
	model.add(Flatten())
	#model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	'''
	model.compile(loss=keras.losses.categorical_crossentropy,
		optimizer=keras.optimizers.Adadelta(),
		metrics=['accuracy'])
	'''
	model.compile(loss=keras.losses.categorical_crossentropy,
		optimizer=optimizers.SGD(lr=0.01),
		metrics=['accuracy'])

	# history
	history = LossHistory()

	model.fit(x_train, y_train,
		batch_size=batch_size,
		epochs=epochs,
		verbose=1,
		validation_data=(x_test, y_test),
		callbacks=[history])
	score = model.evaluate(x_test, y_test, verbose=1)
	model.save('single_polarity.cnn')
	weights = model.layers[0].get_weights()
	plot_model(model, to_file='model_plot.png')
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	history.loss_plot('epoch')

	# stats

	loss1=history.losses['epoch']+history.val_loss['epoch']+history.accuracy['epoch']+history.val_acc['epoch']
	np.savetxt("loss_single.txt",loss1)

	time_end = time.time()

	time_c= time_end - time_start
	print('time cost', time_c, 's')

	# data plots
	# result = sub_z.flatten()
	# peaks, _ = find_peaks(result, distance=20)
	# peaks2, _ = find_peaks(result, prominence=1)
	# peaks3, _ = find_peaks(result, width=20)
	# peaks4, _ = find_peaks(result, threshold=0.4)  # Required vertical distance to its direct neighbouring samples
	# plt.subplot(2, 2, 1)
	# plt.plot(peaks, result[peaks], "xr"); plt.plot(result); plt.legend(['distance'])
	# plt.subplot(2, 2, 2)
	# plt.plot(peaks2, result[peaks2], "ob"); plt.plot(result); plt.legend(['prominence'])
	# plt.subplot(2, 2, 3)
	# plt.plot(peaks3, result[peaks3], "vg"); plt.plot(result); plt.legend(['width'])
	# plt.subplot(2, 2, 4)
	# plt.plot(peaks4, result[peaks4], "xk"); plt.plot(result); plt.legend(['threshold'])
	# plt.show()

	# find peaks without cnn

	# indexes, _ = scipy.signal.find_peaks(sub_z[0],height=1.2, distance=2.1)
	# print('Peaks with minimum height and distance filters: %s' % (indexes))
	# #print('Difference %s', sub_z[0,1515]-sub_z[0,1516])

	# test = sub_z[1][50]
	# print(test)
	# if test > 0:
	#	print('Positive?')
	# else:
	#	print('Negative?')

	# indexes, _ = scipy.signal.find_peaks(sub_z(0))
	# print('Peaks without any filters: {}'.format(indexes))
	# prominences = peak_prominences(vector, indexes)[0]
	# print(prominences)
