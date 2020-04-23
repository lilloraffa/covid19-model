import pandas as pd
import numpy as np
import pickle
from .utils import *

def predNextDays(optmod_name, opt_mod, var_name, pred_days):
    pred = (opt_mod[optmod_name]['mod_data'][var_name])[opt_mod[optmod_name]['i_start'] + opt_mod[optmod_name]['period'] -1 :opt_mod[optmod_name]['i_start'] + opt_mod[optmod_name]['period']+pred_days]
    print("Mod: %s \t Next days: %s: \t %s" %(optmod_name, var_name, str([int(x) for x in pred])))
    print("Mod: %s \t Variation: %s: \t %s" %(optmod_name, var_name, str([int(x) for x in (pred[1:len(pred)] - pred[0:len(pred)-1])])))
        

class ModelStats:
    def __init__(self, model, act_data, pred_days = 10):
        self.model = model
        self.act_data = act_data
        self.data = pd.DataFrame(self.calcData())
        self.data.set_index("date", inplace=True)
        
        
    def printKpi(self, date, kpi_name, title, num_format = 'd', bperc = False):
        var_uff = "uff_" + kpi_name
        var_mod = "mod_" + kpi_name
        if "uff_" + kpi_name in self.data.columns.tolist():
            
            #print(("%30s: %7" + num_format + " vs %7" + num_format + " (%5" + num_format + " vs %5" + num_format + "), errore: %" + num_format + "") %(
            print(("%30s: %7s vs %7s (%5s vs %5s), errore: %s") %(
                title,
                format_number(self.data[var_uff][np.datetime64(date, 'D')], bperc = bperc),
                format_number(self.data[var_mod][np.datetime64(date, 'D')], bperc = bperc),
                format_number(self.data[var_uff][np.datetime64(date, 'D')] - self.data[var_uff][np.datetime64(date, 'D') - np.timedelta64(1, 'D')], bperc = bperc),
                format_number(self.data[var_mod][np.datetime64(date, 'D')] - self.data[var_mod][np.datetime64(date, 'D') - np.timedelta64(1, 'D')], bperc = bperc),
                format_number(self.data[var_uff][np.datetime64(date, 'D')] - self.data[var_mod][np.datetime64(date, 'D')], bperc = bperc)
            ))
        else:
            #print(("%30s: %7" + num_format + " (%5" + num_format + ")") %(
            print(("%30s: %7s (%5s)") %(
                title,
                format_number(self.data[var_mod][np.datetime64(date, 'D')], bperc = bperc),
                format_number(self.data[var_mod][np.datetime64(date, 'D')] - self.data[var_mod][np.datetime64(date, 'D') - np.timedelta64(1, 'D')], bperc = bperc)
            ))
    
    def printKpis(self, date):
        self.printKpi(date, 'Igc_cum', "Tot Infected")
        self.printKpi(date, 'Igc', "Currently Infected")
        self.printKpi(date, 'Igci_t', "Currently in Int. Care")
        self.printKpi(date, 'Gc_cum', "Tot Recovered")
        self.printKpi(date, 'M_cum', "Tot Dead")
        print()
        self.printKpi(date, 'Igc_cum_pinc', "% Increase, Infected", num_format=".3f", bperc = True)
        self.printKpi(date, 'ratio_Gc_Igc', "% Mortality Rate", num_format=".3f", bperc = True)
        self.printKpi(date, 'ratio_M_Igc', "% Known Recovery Rate", num_format=".3f", bperc = True)
        print()
        self.printKpi(date, 'ratio_Gccum_Igccum', "% Recovered / Tot", num_format=".3f", bperc = True)
        self.printKpi(date, 'ratio_Mcum_Igccum', "% Dead / Tot", num_format=".3f", bperc = True)
        self.printKpi(date, 'ratio_Igci_Igc', "% Intensive Care", num_format=".3f", bperc = True)
        self.printKpi(date, 'ratio_Igcn_Igc', "% Non Intensive Care", num_format=".3f", bperc = True)
        self.printKpi(date, 'ratio_I_Igc', "% Total Infected / Known Infected", num_format=".3f", bperc = True)
        self.printKpi(date, 'R0_t', "R0", num_format=".3f")
        print()
        print()
        print("*** 7 days ahead predictions ***")
        self.printPredict(date, 'Igc_cum', "Tot Infettati", pred_step = 7, bperc = False)
        print()
        self.printPredict(date, 'Igc', "Attualmente Infetti", pred_step = 7, bperc = False)
        print()
        self.printPredict(date, 'Igci_t', "Attualmente in Intensiva", pred_step = 7, bperc = False)
        print()
        self.printPredict(date, 'Gc_cum', "Tot Guariti", pred_step = 7, bperc = False)
        print()
        self.printPredict(date, 'M_cum', "Tot Morti", pred_step = 7, bperc = False)
    
    
    def printPredict(self, curr_date, kpi_name, title, pred_step = 7, bperc = False):
        
        var_mod = "mod_" + kpi_name
        
        data = self.data[var_mod][np.datetime64(curr_date, 'D') : np.datetime64(np.datetime64(curr_date, 'D') + np.timedelta64(pred_step, 'D'))]
        data_delta = pd.Series(data).diff(1)
        data_str = "["
        for val in data:
            data_str = " " + data_str + " {:7s}".format(format_number(val)) + " "
        data_str = data_str + "]"
        
        data_delta_str = "["
        for val in data_delta:
            #data_delta_str = " " + data_delta_str + " {:7s}".format(format_number(val)) + " "
            #print(val)
            #if math.isfinite(val):
            data_delta_str = " " + data_delta_str + " {:7s}".format(str(format_number(val))) + " "
            #else:
            #    data_delta_str = " " + data_delta_str + " {:7s}".format("0") + " "
        data_delta_str = data_delta_str + "]"

        print(("%30s: %60s") %(
                title,
                data_str
            ))
        print(("%30s: %60s") %(
                "Var.",
                data_delta_str
            ))
    
        
    def calcData(self):
        def calcDataVar(data):
            istart = self.model['i_start']
            #iend = istart + len(data)
            mod_len = len(self.model['mod_data']['dat_Igc']) 
            #return [np.NaN for i in range (0, istart)] + data.tolist() + [np.NaN for i in range(istart + len(data) -1, mod_len-1)]
            return [np.NaN for i in range (0, istart)] + data.tolist()[self.act_data.i_start:] + [np.NaN for i in range(istart + len(data[self.act_data.i_start:]) -1, mod_len-1)]
        
        def calcDataVarDate(data):
            istart = self.model['i_start']
            mod_len = len(self.model['mod_data']['dat_Igc'])
            #first_date = data[0] - np.timedelta64(istart, 'D')
            first_date = data[self.act_data.i_start] - np.timedelta64(istart, 'D')
            return [np.datetime64(first_date + np.timedelta64(i, 'D'), 'D') for i in range (0, mod_len)]
        
        
        uff_Igci_t = calcDataVar(self.act_data.dat_Igci_t)
        uff_Igcn_t = calcDataVar(self.act_data.dat_Igcn_t)
        uff_Igc = calcDataVar(self.act_data.dat_Igc)
        uff_Igc_cum = calcDataVar(self.act_data.dat_Igc_cum)
        uff_Gc_cum = calcDataVar(self.act_data.dat_Gc_cum)
        uff_M_cum = calcDataVar(self.act_data.dat_M_cum)
        uff_Gc = [np.NaN] + np.diff(uff_Gc_cum).tolist()
        uff_M = [np.NaN] + np.diff(uff_M_cum).tolist()
        uff_Igc_cum_pinc = (pd.Series(uff_Igc_cum)/pd.Series(uff_Igc_cum).shift(1)) - 1
        uff_ratio_Gc_Igc = (pd.Series(uff_Gc)/pd.Series(uff_Igc).shift(1))
        uff_ratio_M_Igc = (pd.Series(uff_M)/pd.Series(uff_Igc).shift(1))
        uff_ratio_Gccum_Igccum = (np.array(uff_Gc_cum)/np.array(uff_Igc_cum)).tolist()
        uff_ratio_Mcum_Igccum = (np.array(uff_M_cum)/np.array(uff_Igc_cum)).tolist()
        uff_ratio_Igci_Igc = (np.array(uff_Igci_t)/np.array(uff_Igc)).tolist()
        uff_ratio_Igcn_Igc = (np.array(uff_Igcn_t)/np.array(uff_Igc)).tolist()
        mod_Igci_t = self.model['mod_data']['dat_Igci_t']
        mod_Igcn_t = self.model['mod_data']['dat_Igcn_t']
        mod_Ias_t = self.model['mod_data']['dat_Ias_t']
        mod_Igs_t = self.model['mod'].Igs_t
        mod_Igc = self.model['mod_data']['dat_Igc']
        mod_Igc_cum = self.model['mod_data']['dat_Igc_cum']
        mod_I = self.model['mod_data']['dat_I']
        #mod_NIs_t = self.model['mod_data']['dat_NIs']
        mod_G = self.model['mod_data']['dat_G']
        mod_Gc = self.model['mod_data']['dat_Gc']
        mod_M = self.model['mod_data']['dat_M']
        mod_G_cum = self.model['mod_data']['dat_G_cum']
        mod_Gc_cum = self.model['mod_data']['dat_Gc_cum']
        mod_M_cum = self.model['mod_data']['dat_M_cum']
        mod_Popi_t = self.model['mod_data']['dat_Popi_t']
        mod_R0_t = self.model['mod_data']['dat_R0_t']
        mod_Igc_cum_pinc = (pd.Series(mod_Igc_cum)/pd.Series(mod_Igc_cum).shift(1)) - 1
        mod_ratio_M_Igc = (pd.Series(mod_M)/pd.Series(mod_Igc).shift(1))
        mod_ratio_Gc_Igc = (pd.Series(mod_Gc)/pd.Series(mod_Igc).shift(1))
        mod_ratio_Gccum_Igccum = (np.array(mod_Gc_cum)/np.array(mod_Igc_cum)).tolist()
        mod_ratio_Mcum_Igccum = (np.array(mod_M_cum)/np.array(mod_Igc_cum)).tolist()
        mod_ratio_Igci_Igc = (np.array(mod_Igci_t)/np.array(mod_Igc)).tolist()
        mod_ratio_Igcn_Igc = (np.array(mod_Igcn_t)/np.array(mod_Igc)).tolist()
        mod_ratio_Ias_Igc = (np.array(mod_Ias_t)/np.array(mod_Igc)).tolist()
        mod_ratio_I_Igc = ((np.array(mod_Ias_t) + np.array(mod_Igs_t) + np.array(mod_Igc))/np.array(mod_Igc)).tolist()
        
        res =  {
            'date': calcDataVarDate(self.act_data.date),
            'uff_Igci_t': calcDataVar(self.act_data.dat_Igci_t),
            'uff_Igcn_t': calcDataVar(self.act_data.dat_Igcn_t),
            'uff_Igc': calcDataVar(self.act_data.dat_Igc),
            'uff_Igc_cum': calcDataVar(self.act_data.dat_Igc_cum),
            'uff_Gc_cum': calcDataVar(self.act_data.dat_Gc_cum),
            'uff_M_cum': calcDataVar(self.act_data.dat_M_cum),
            'uff_Gc': uff_Gc, 
            'uff_M': uff_M,
            'uff_ratio_Igci_Igc': uff_ratio_Igci_Igc,
            'uff_ratio_Igcn_Igc': uff_ratio_Igcn_Igc,
            'uff_Igc_cum_pinc': uff_Igc_cum_pinc,
            'uff_ratio_Gc_Igc': uff_ratio_Gc_Igc,
            'uff_ratio_M_Igc': uff_ratio_M_Igc,
            'uff_ratio_Gccum_Igccum': uff_ratio_Gccum_Igccum,
            'uff_ratio_Mcum_Igccum': uff_ratio_Mcum_Igccum,
            'mod_Igci_t' : self.model['mod_data']['dat_Igci_t'],
            'mod_Igcn_t' : self.model['mod_data']['dat_Igcn_t'],
            'mod_Ias_t' : self.model['mod_data']['dat_Ias_t'],
            'mod_Igc' : self.model['mod_data']['dat_Igc'],
            'mod_Igc_cum' : self.model['mod_data']['dat_Igc_cum'],
            'mod_I' : self.model['mod_data']['dat_I'],
            #'mod_NIs' : self.model['mod'].NIs_t,
            'mod_G' : self.model['mod_data']['dat_G'],
            'mod_Gc' : self.model['mod_data']['dat_Gc'],
            'mod_M' : self.model['mod_data']['dat_M'],
            'mod_G_cum' : self.model['mod_data']['dat_G_cum'],
            'mod_Gc_cum' : self.model['mod_data']['dat_Gc_cum'],
            'mod_M_cum' : self.model['mod_data']['dat_M_cum'],
            'mod_Popi_t' : self.model['mod_data']['dat_Popi_t'],
            'mod_R0_t' : mod_R0_t,
            'mod_Igc_cum_pinc': mod_Igc_cum_pinc,
            'mod_ratio_Gc_Igc': mod_ratio_Gc_Igc,
            'mod_ratio_M_Igc': mod_ratio_M_Igc,
            'mod_ratio_Gccum_Igccum': mod_ratio_Gccum_Igccum,
            'mod_ratio_Mcum_Igccum': mod_ratio_Mcum_Igccum,
            'mod_ratio_Igci_Igc': mod_ratio_Igci_Igc,
            'mod_ratio_Igcn_Igc': mod_ratio_Igcn_Igc,
            'mod_ratio_Ias_Igc': mod_ratio_Ias_Igc,
            'mod_ratio_I_Igc': mod_ratio_I_Igc,
        }
        
        return res