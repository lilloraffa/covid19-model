import pandas as pd
import numpy as np
from datetime import datetime
import dateutil
import requests

class ImportData:
    def __init__(self, ext_csv, ext_csv_anagr = 'data_regioni_anagrafica.csv', rel_path = ''):
        self.etx_csv = ext_csv
        self.data_orig = pd.read_csv(rel_path + ext_csv)
        self.data = pd.read_csv(rel_path + ext_csv).groupby(['date', 'geo_code']).sum().reset_index()
        self.data_anagr = pd.read_csv(rel_path + ext_csv_anagr)
        self.data['date'] = self.data['date'].apply(dateutil.parser.parse)
        self.data_aggr = self.calcAggrData(self.data)

    def calcAggrData(self, data):
        return data.groupby(by='date').sum().reset_index()

class ActualData:
    def __init__(self, data):
        self.date = np.asarray(data['date'])
        self.dat_Igci_t = np.asarray(data['inf_hosp_intensive'])
        self.dat_Igcn_t = np.asarray(data['inf_hosp_nonintensive']) + np.asarray(data['inf_home'])
        self.dat_Igc = self.dat_Igci_t + self.dat_Igcn_t
        self.dat_Gc_cum = np.asarray(data['inf_recovered'])
        self.dat_M_cum = np.asarray(data['inf_dead'])
        self.dat_Igc_cum = np.asarray(data['inf_tot_cum'])
        self.i_start = self.getStartTime()
        self.len = len(data['date'])
        self.startDate = data['date'][self.i_start]
        self.endDate = data['date'][self.len-1]
        
    def getStartTime(self):
        start = np.where(self.dat_Igc_cum>5)[0]
        return start[0] if len(start)>0 else None

def getItaData(rel_path = ''):
    
    url = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv'
    r = requests.get(url, allow_redirects=True)

    open('data/sources/dpc-covid19-ita-regioni.csv', 'wb').write(r.content)
    
    
    data_org = pd.read_csv(rel_path + "data/sources/dpc-covid19-ita-regioni.csv").groupby(['data', 'codice_regione']).sum().reset_index()
    data = pd.DataFrame(
        {
            'date': data_org['data'],
            'geo_code': data_org['codice_regione'],
            #'geo_name': data_org['denominazione_regione'],
            'inf_hosp_intensive': data_org['terapia_intensiva'],
            'inf_hosp_nonintensive': data_org['ricoverati_con_sintomi'],
            'inf_home': data_org['isolamento_domiciliare'],
            'inf_recovered': data_org['dimessi_guariti'],
            'inf_dead': data_org['deceduti'],
            'inf_tot_cum': data_org['totale_casi']
        }
    )
    data.to_csv(rel_path + "data/covid_ita_regional.csv")
    
    data_anagr_org = pd.read_csv(rel_path + "data/sources/data_regioni_anagrafica.csv", sep=';')
    data_anagr = pd.DataFrame(
        {
            'geo_code': data_anagr_org['codice_regione'],
            'geo_name': data_anagr_org['denominazione_regione'],
            'pop': data_anagr_org['pop'],
            'area': data_anagr_org['superficie'],
            'density': data_anagr_org['densita'],
            'n_city': data_anagr_org['num_comuni']
        }
    )
    data_anagr.to_csv(rel_path + "data/anagr_ita_regional.csv")

def refreshData(rel_path = ''):
    getItaData(rel_path = rel_path)