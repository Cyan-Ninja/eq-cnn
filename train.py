"""
Predict polarity using find peaks of scipy

https://wiki.seg.org/wiki/Phase_and_polarity_assessment_of_seismic_data#Types_of_Polarity

	American polarity: positive polarity (impedance) is linked to a peak (positive amplitude)

	European polarity: opposite of the American one, which means a positive polarity (impedance)
	 associated with a trough (negative amplitude)
"""
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
from typing import Union, Optional, List, Mapping, Dict, Any
#from vector import vector, plot_peaks

import matplotlib.pyplot as plt
import numpy as np
import scipy
import obsplus
import obspy
import pandas as pd
import pytest
import tables
from scipy.signal import find_peaks,peak_prominences
import obsplus
from obsplus.utils.time import to_utc

#@dataclass
#class PeakPolarityGetter:
	#"""Class for finding polarity of an array."""
	# def process(self, array: np.ndarray, pick_index: np.ndarray):
	#	assert len(array) == len(pick_index)


def read_data(path, name) -> np.array:
	"""
	Read the data (time series) arrays

	Dimension meaning is as follows:

		dim0 - trace
		dim1 - time
		dim2 - channel (order is Z E N, use only Z)


	"""
	with tables.open_file(dataset_path) as store:
		out = store.root[name]['data'][:]
		# print datasets print(store.root)
	return out

def read_labels(path, name) -> pd.DataFrame:
	"""Read the labels of the dataset"""
	df = pd.read_hdf(path, f"{name}/index")
	return df

if __name__ == "__main__":
	# define variables
	dataset_path = 'dataset.h5'
	dataset_name = 'R2'
	num_traces_to_plot = 5

	# read data and labels
	data = read_data(dataset_path, dataset_name)
	df = read_labels(dataset_path, dataset_name)

	# slice data/labels and plot
	sub_data = data[:num_traces_to_plot]  #Data
	df_sub = df.iloc[:num_traces_to_plot] #labels

	sub_z = data[:, :, 0] # just grab z component sub_data[:, :, 0]

	# example on data
	print('Data: {}'.format(data[0]))
	print('Time: {}'.format(df['time'][0]))
	print('Polarity: {}'.format(df['polarity'][0]))
	print('Z-Only: {}'.format(sub_z[0]))
	print('Time: {}'.format(df['time'][0]))


	# # add sample rate
	# def _get_pick_samples(df):
	# 	 """get the pick sample relative to start of trace"""
	# 	 dt = (df['time'] - df['starttime']).dt.total_seconds()
	# 	 out = dt * df['sampling_rate']
	# 	 return out
	#
	# df['pick_sample'] =_get_pick_samples(df)

	# plots
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

	# get result from model

	# BLAH BLAH BLAH MAGIC

	# find peaks without cnn

	# indexes, _ = scipy.signal.find_peaks(sub_z[0],height=1.2, distance=2.1)
	# print('Peaks with minimum height and distance filters: %s' % (indexes))
	# #print('Difference %s', sub_z[0,1515]-sub_z[0,1516])

	# test = sub_z[1][50]
	# print(test)
	# if test > 0:
	# 	print('Positive?')
	# else:
	# 	print('Negative?')

	# indexes, _ = scipy.signal.find_peaks(sub_z(0))
	# print('Peaks without any filters: {}'.format(indexes))
	# prominences = peak_prominences(vector, indexes)[0]
	# print(prominences)
