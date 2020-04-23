import sys
import os
import pickle
import pandas as pd
import numpy as np
import math
from datetime import datetime
import dateutil
import copy
import matplotlib.pyplot as plt
#from sklearn.linear_model import LogisticRegression, LinearRegression


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

def format_number(num, bperc = False):
    def format_units(num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        # add more suffixes if you need them
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    
    def format_perc(num):
        return '%.2f' %(num * 100)
        
    if math.isfinite(num):
        if bperc: 
            return format_perc(num)
        else:
            return format_units(num)
    else: "%s" %(num)


class ImportDataOld:
    def __init__(self, ext_csv, reg_anagrafica = 'data_regioni_anagrafica.csv', rel_path = ''):
        self.etx_csv = ext_csv
        self.data_regioni_orig = pd.read_csv(rel_path + ext_csv)
        self.data_regioni = pd.read_csv(rel_path + ext_csv).groupby(['data', 'codice_regione']).sum().reset_index()
        self.anagr_regioni = pd.read_csv(rel_path + reg_anagrafica, sep=';')
        self.data_regioni['data'] = self.data_regioni['data'].apply(dateutil.parser.parse)
        self.data_nazionale = self.calc_NationalData(self.data_regioni)

    def calc_NationalData(self, data):
        return data.groupby(by='data').sum().reset_index()

# Get the id and value of the element of the array closest to value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (idx, array[idx])

