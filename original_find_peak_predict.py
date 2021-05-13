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

import local  # This is just a file for storing common vars
from local import output_cache
from local import hdf_out

#@dataclass
#class PeakPolarityGetter:
   #"""Class for finding polarity of an array."""
   # def process(self, array: np.ndarray, pick_index: np.ndarray):
   #   assert len(array) == len(pick_index)


def read_data(path, name) -> np.array:
    """
    Read the data (time series) arrays
    
    Dimension meaning is as follows:

        dim0 - trace
        dim1 - time
        dim2 - channel (order is Z E N)

    
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
    dataset_path = local.hdf_out
    dataset_name = 'R2'
    num_traces_to_plot = 5

    # read data and labels
    data = read_data(dataset_path, dataset_name)
    df = read_labels(dataset_path, dataset_name)

    # slice data/labels and plot
    sub_data = data[:num_traces_to_plot]  #Data
    df_sub = df.iloc[:num_traces_to_plot] #labels
  

    print(sub_data[0])
    result = sub_data.flatten()  # just grab z component sub_data[:, :, 0]
    altTest = sub_data[:, :, 0]
    print(result[0])

    def _get_pick_samples(df):
        """get the pick sample relative to start of trace"""
        dt = (df['time'] - df['starttime']).dt.total_seconds()
        out = dt * df['sampling_rate']
        return out

    #df['pick_sample'] =_get_pick_samples(df)

    breakpoint()
    # tests
      
    peaks, _ = find_peaks(result, distance=20)
    peaks2, _ = find_peaks(result, prominence=1)      
    peaks3, _ = find_peaks(result, width=20)
    peaks4, _ = find_peaks(result, threshold=0.4)  # Required vertical distance to its direct neighbouring samples
    plt.subplot(2, 2, 1)
    plt.plot(peaks, result[peaks], "xr"); plt.plot(result); plt.legend(['distance'])
    
    plt.subplot(2, 2, 2)
    plt.plot(peaks2, result[peaks2], "ob"); plt.plot(result); plt.legend(['prominence'])
    
    plt.subplot(2, 2, 3)
    plt.plot(peaks3, result[peaks3], "vg"); plt.plot(result); plt.legend(['width'])
    
    plt.subplot(2, 2, 4)
    plt.plot(peaks4, result[peaks4], "xk"); plt.plot(result); plt.legend(['threshold'])
    
    #plt.show()
    

    #vector = np.array([0, 6, 25, 20, 15, 8, 15, 6, 0, 6, 0, -5, -15, -3, 4, 10, 8,
                   #13, 8, 10, 3, 1, 20, 7, 3, 0])

    vector = result

    print('Detect peaks with minimum height and distance filters.')
    indexes, _ = scipy.signal.find_peaks(altTest[0],height=1.2, distance=2.1)
    breakpoint()
  
    print('Peaks are: %s' % (indexes))
    #print('Diffrence %s', altTest[0,1515]-altTest[0,1516])

    test = altTest[0,indexes[0]]-altTest[0,indexes[0]+1]

    if test > 0:
     print('positive?')
    else:
     print('netgitive?')
    
    print('Detect peaks without any filters.')

    indexes, _ = scipy.signal.find_peaks(np.array(vector))
    print('Peaks are: {}'.format(indexes))
    
    prominences = peak_prominences(vector, indexes)[0]
    print(prominences)


