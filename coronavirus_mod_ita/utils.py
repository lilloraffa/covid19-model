import sys
import pickle
import pandas as pd
import numpy as np
import math
from datetime import datetime
import dateutil
import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

class ImportData:
    def __init__(self, ext_csv, reg_anagrafica = 'data_regioni_anagrafica.csv'):
        self.etx_csv = ext_csv
        self.data_regioni_orig = pd.read_csv(ext_csv)
        self.data_regioni = pd.read_csv(ext_csv).groupby(['data', 'codice_regione']).sum().reset_index()
        self.anagr_regioni = pd.read_csv(reg_anagrafica, sep=';')
        self.data_regioni['data'] = self.data_regioni['data'].apply(dateutil.parser.parse)
        self.data_nazionale = self.calc_NationalData(self.data_regioni)

    def calc_NationalData(self, data):
        return data.groupby(by='data').sum().reset_index()

def saveOptMod(obj, actual_data_update, path = "", mod_name="generic", bDataUpdateTime = True, bRunDatetime = True):
    
    #data_uff.date[len(data_uff.date)-1]
    actual_data_update = np.datetime_as_string(actual_data_update, unit='D')
    current_date = np.datetime_as_string(np.datetime64(datetime.now()), unit='s')
    path_full = path + mod_name
    if bDataUpdateTime: path_full = path_full + "_act-" + actual_data_update
    if bRunDatetime: path_full = path_full + "_" + current_date 
    with open(path_full, 'wb') as saved_file:
        pickle.dump(obj, saved_file)

# Get the id and value of the element of the array closest to value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (idx, array[idx])

