import sys
import pickle
import pandas as pd
import numpy as np
import math
from datetime import datetime
import dateutil
import copy
from .model import * 
from .utils import *
from .grid_management import *

class ModelOptimizer:
    dic_mod_name = {
        'tot': 2,
        'Igc_cum': 3,
        'Igc': 4,
        'Gc_cum': 5,
        'M_cum': 6
        #'Igci_t': 9
    }
    def __init__(self, act_data, grid, model_stepsforward, sync_data_perc = 0.1, window_len = 7, err_perc_tres = 0.001, 
                Pop_tot = 50000000,
                opt_model = None,
                opt_model_window = None,
                exclude_param_finetuner = None,
                exclude_param_window = None):

        self.act_data = act_data
        self.grid = grid
        self.sync_data_perc = sync_data_perc
        self.model_stepsforward = model_stepsforward
        self.model_runs = list()
        self.window_len = window_len
        self.Pop_tot = Pop_tot
        self.err_perc_tres = err_perc_tres
        self.exclude_param_finetuner = exclude_param_finetuner
        self.exclude_param_window = exclude_param_window

        if opt_model is not None:
            self.opt_model = opt_model
            self.opt_model_initial = opt_model
        else:
            self.opt_model = {}
            self.opt_model_initial = None
        
        self.opt_model_window = opt_model_window
        
    
    def start(self):
        initial_time = datetime.now()
        print("**** Model optimization started at: " + str(initial_time))
        self.gridOptimizer()
        final_time = datetime.now()
        print("**** Model optimization ended at: " + str(final_time))
        print("\t in: " + str(final_time - initial_time))

        print()
        print("******** Initial Models *********")
        for model_name in self.opt_model.keys():
            self.printModelSel(model_name, self.opt_model[model_name], bprint=True)
        print()
        initial_time_new = final_time
        print("**** Model AllOptParamFinetuner started at: " + str(initial_time_new))
        #model_names = ['tot', 'Igc_cum', 'Igc', 'Gc_cum', 'M_cum']
        model_names = ['tot']

        finetuned_params, ex = getParamList(exclude = (self.exclude_param_finetuner + ['rg_period', 'ra_period']))
        self.opt_model = self.gridAllOptParamFinetuner(self.opt_model, model_names, finetuned_params = finetuned_params, deltas = [0.1, 0.05, 0.01], grid_steps = 2, opt_max_iter = 100, bprint = True)
        if self.opt_model_window is not None:
            self.opt_model_window = self.gridAllOptParamFinetuner(self.opt_model_window, model_names, finetuned_params = finetuned_params, deltas = [0.1, 0.05, 0.01], grid_steps = 2, opt_max_iter = 100, bprint = True)


        initial_time_new = final_time
        print("**** Model WindowOptimizer started at: " + str(initial_time_new))
        initial_time_new = final_time
        model_names = ['tot']

        window_params, ex = getParamList(param_list_init = ['rg_period', 'ra_period', 'alpha', 'beta', 'beta_gcn', 'gamma'], exclude = (self.exclude_param_window))
        finetuned_params, ex = getParamList(exclude = (self.exclude_param_finetuner + ['rg', 'ra', 'rg_period', 'ra_period']))
        self.windowParamOptimizer(model_names, self.window_len, delta_perc = 0.01, opt_max_iter = 100, window_params = window_params, finetuned_params = finetuned_params, bOptFinetuner = False, bprint=True)

        final_time = datetime.now()
        print("**** Model WindowOptimizer ended at: " + str(final_time))
        print("\t in: " + str(final_time - initial_time_new))
        
        print()
        print("******** Optimal Models *********")
        for model_name in self.opt_model.keys():
            self.printModelSel(model_name, self.opt_model[model_name], bprint=True)
        
        if self.opt_model_window is not None:
            print()
            print("******** Optimal Window Models *********")
            for model_name in self.opt_model.keys():
                self.printModelSel(model_name, self.opt_model_window[model_name], bprint=True)
        
        print("**** Optimization Procedure finished in: %s" %(str(final_time - initial_time)))

    def printModelSel(self, model_name, model, bprint=True):
        if bprint:
            print("Opt. model: %s \t (%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %d, %d, %d, %d, %d, %d) \n\t err_tot= %s \t err_Igc_cum= %s \t err_G= %s \t err_M= %s" %(
                model_name,
                model['mod'].rg,
                model['mod'].ra,
                model['mod'].alpha,
                model['mod'].beta,
                model['mod'].beta_gcn,
                model['mod'].gamma, 
                model['mod'].t1,
                model['mod'].tgi2,
                model['mod'].tgn2,
                model['mod'].ta2,
                model['mod'].Igs_t0,
                model['mod'].Ias_t0,     
                format_number(model['err_tot']),
                format_number(model['err_Igc_cum']),
                format_number(model['err_Gc_cum']),
                format_number(model['err_M_cum'])
            ))
        
    
    def optModelSel(self, mod_name, bprint=False):
        mod_sel_i = np.nanargmin(np.asarray(self.model_runs)[:,self.dic_mod_name[mod_name]].astype(float))
        return self.model_runs[mod_sel_i][8]

    def optModels(self, mod_names, bprint=False):
        for name in mod_names:
            self.opt_model[name] = self.optModelSel(name, bprint)
    
    def memorySaver(self, num_models, bprint=False):
        initial_time = datetime.now()
        model_runs_new = list()
        if bprint: print("\t memorySaver started at: %s - " %(str(initial_time)), end="")
        for i in self.dic_mod_name.values():
            self.model_runs.sort(key = lambda x:x[i])
            model_runs_new = model_runs_new + self.model_runs[0:num_models].copy()
        if bprint: print("MemorySaver - before %d \t " %(get_size(self.model_runs)), end="")
        self.model_runs = model_runs_new.copy()
        del model_runs_new
        if bprint: print("after %d \t " %(get_size(self.model_runs)))
        if bprint: print("finished in: %s" %(str(datetime.now() - initial_time)))

    def gridAllOptParamFinetuner(self, optmodels_in, model_names, finetuned_params = None, deltas = [0.1, 0.05, 0.01], grid_steps = 2, opt_max_iter = 100, bprint = False, err_perc_tres = 0.01):
        initial_time = datetime.now()
        print("****** Started OptParamFinetuner ******", model_names, finetuned_params)
        mod_generic = Model()
        optmodels = optmodels_in.copy()
        
        #initialize parameters to be finetuned
        params = {}
        for param in mod_generic.params.keys():
            params[param] = 1
        
        #calculate which parameters needs to be finetuned
        finetuned_params_final = []
        if finetuned_params is not None:
            finetuned_params_final = finetuned_params
        else:
            finetuned_params_final = ['rg', 'ra', 'alpha', 'beta', 'beta_gcn', 'gamma', 't1', 'tgi2', 'tgn2', 'ta2', 'Igs_t0', 'Ias_t0']
            
        for param in finetuned_params_final:
                params[param] = grid_steps
                
        
        for model_name in model_names:
            bIter = True
            iIter = 0
            iIterCycle = 0
            iIterCycle2 = 0
            
            while bIter and iIter < opt_max_iter :
                bIter = False
                iIter = iIter + 1
                grid_param = GridParam()
                model = optmodels[model_name]['mod']
                delta_perc = deltas[iIterCycle]
                
                for param in params.keys():
                    param_val = getattr(model, param)
                    
                    if param in ['t1', 'tgi2', 'tgn2', 'ta2']:
                        delta_inc = 1
                        if delta_perc >= 0.1:
                            delta_inc = 3
                        elif delta_perc >= 0.05 and delta_perc < 0.1:
                            delta_inc = 2

                        param_min = 2 if param_val <= 2 else max(param_val - delta_inc, 2)
                        grid_param.setGrid(Param(param, par_min = 2, par_max=40), grid_avg = param_val, grid_min = param_min, grid_max = min(param_val + delta_inc, 40), steps = params[param])
                    elif param in ['rg_period', 'ra_period']:     
                        grid_param.setGridList(Param(param), [param_val])
                    else: 
                        if param in ['alpha', 'gamma', 'beta', 'beta_gcn']:
                            if param_val >= 1: param_val = 0.9999
                            elif param_val <=0: param_val = 0.00000001
                        grid_param.setGrid(Param(param), grid_avg = param_val, grid_min = (param_val*(1-delta_perc)), grid_max = (param_val*(1+delta_perc)), steps = params[param])

                #Find the optimal models with new grid
                mod_optimizer = ModelOptimizer(self.act_data, grid_param, self.model_stepsforward, sync_data_perc = self.sync_data_perc, Pop_tot = self.Pop_tot)
                mod_optimizer.gridOptimizer()

                err_prev = optmodels[model_name]["err_" + model_name]
                err_new = mod_optimizer.opt_model[model_name]["err_" + model_name]
                err_delta = err_prev if err_new == 0.0 else (err_prev - err_new)/err_new
            
                if err_delta > err_perc_tres:
                    optmodels[model_name] = mod_optimizer.opt_model[model_name].copy()
                    mod = optmodels[model_name]['mod']
                    bIter = True
                    if bprint: print("\t (%d) Found FineTuner: %s (delta: %.3f, prev: %s -  new:%s - %.2f)" 
                                        %(iIter, model_name, delta_perc, format_number(err_prev), format_number(err_new), err_new/err_prev - 1.0), end="")
                    if bprint: print(", in %s" %(str(datetime.now() - initial_time).split(".")[0]))
                    if bprint: print("\t\t m:(%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %d, %d, %d, %d, %.1f, %.1f)" 
                                        %(mod.rg, mod.ra, mod.alpha, mod.beta, mod.beta_gcn, mod.gamma, mod.t1, mod.tgi2, mod.tgn2, mod.ta2, mod.Igs_t0, mod.Ias_t0))
                else:
                    iIterCycle = iIterCycle + 1
                    if iIterCycle < len(deltas): 
                        bIter = True
                    elif iIterCycle2 < 1 and len(deltas)>1:
                        iIterCycle2 = 1
                        iIterCycle = 0
                        bIter = True

        return optmodels


    def calcGrid(self, model, param_list = [], delta_perc = 0.01, window_curr = None, window_len = None, steps = 3):
        params = model.params
        grid_param = GridParam()

        if(window_curr is None or window_len is None):
            window_curr = 0
            window_len = len(model.Ias_t)


        for param in params.keys():
            param_val = params[param]
            if param in param_list:
                if str(param) in ['t1', 'tgi2', 'tgn2', 'ta2']:
                    delta_inc = 1
                    if delta_perc >= 0.1:
                        delta_inc = 3
                    elif delta_perc >= 0.05 and delta_perc < 0.1:
                        delta_inc = 2
                    param_min = 2 if param_val <= 2 else max(param_val - delta_inc, 2)
                    grid_param.setGrid(Param(param, par_min = 2, par_max=40), grid_avg = param_val, grid_min = param_min, grid_max = min(param_val + delta_inc, 40), steps = steps)
                elif param in ['rg_period', 'ra_period']:  
                    if param_val is None:
                        #param_value = grid_param.getGrid(param)[0]
                        val = model.params['rg'] if param == 'rg_period' else model.params['ra']
                        tot_len = len(model.Ias_t)
                        param_val = [val for i in range(0, tot_len)]
                    #grid_param.setGridList(Param(param), [param_value])

                    param_min = list(map(lambda x, i: x*(1 - delta_perc) if i in range(window_curr, window_curr + window_len) else x, param_val, [j for j in range(0, len(param_val))]))
                    param_max = list(map(lambda x, i: x*(1 + delta_perc) if i in range(window_curr, window_curr + window_len) else x, param_val, [j for j in range(0, len(param_val))]))
                    param_grid = [param_min, param_val, param_max]
                    grid_param.setGridList(Param(param), param_grid)
                else: 
                    #param_min = (param_val*(1-delta_perc))
                    #param_max = (param_val*(1+delta_perc))
                    if param in ['alpha', 'gamma', 'beta', 'beta_gcn']:
                        if param_val >= 1: param_val = 0.9999
                        elif param_val <=0: param_val = 0.00000001
                    if param_val == 0:
                        param_min = 0
                        param_val = 1
                        param_max = 2
                    else:
                        param_min = (param_val*(1-delta_perc))
                        param_max = (param_val*(1+delta_perc))

                    grid_param.setGrid(Param(param), grid_avg = param_val, grid_min = param_min, grid_max = param_max, steps = steps)
            else:   
                grid_param.setGridList(Param(param), [param_val])

        return grid_param
    
    def windowParamOptimizer(self, model_names, window_len, deltas = [0.1, 0.05, 0.01], delta_perc = 0.1, opt_max_iter = 100, window_params = ['rg_period', 'ra_period'], finetuned_params = None, bOptFinetuner = True, bprint = False, err_perc_tres = 0.01):
        print("****** Started OptWindows ******", model_names, window_params)
        prev_time = datetime.now()
        if self.opt_model_window is None:
            opt_mod_window = self.opt_model.copy()
            self.opt_model_window = self.opt_model.copy()
        else:
            opt_mod_window = self.opt_model_window.copy()
        
        for model_name in model_names:
            
            mod_istart = opt_mod_window[model_name]['i_start']
            window_curr = 0
            window_opt_last = -1
            iIterDelta = 0
            bLastWindowTuned = False
            iIterLastWindowTuned = 0
            iIterTot = 0
            #deltas = [0.1, 0.05, 0.01]

            #Cicla su tutte le window possibili, coerenti con la lunghezza dei dati originari
            while (window_curr - mod_istart + round(window_len/2,0)) < self.act_data.len:
                #print("Win: (%d, %d, %d) - data_uff_len: %d" %(window_curr, window_curr + window_len, mod_istart, self.act_data.len))
                bIter = True
                iIter = 0
                
                iIterDelta = 0
                
                #grid_param = grid_param_init
                #Cicla fino a che trovi miglioramenti
                while bIter and iIter < opt_max_iter :
                    delta_perc = deltas[iIterDelta]
                    model = opt_mod_window[model_name]['mod']
                    bIter = False

                    grid_param = self.calcGrid(model, param_list = window_params, delta_perc = delta_perc, window_curr = window_curr, window_len = window_len)

                    #Find the optimal models with new grid
                    mod_optimizer = ModelOptimizer(self.act_data, grid_param, self.model_stepsforward, sync_data_perc = self.sync_data_perc, Pop_tot = self.Pop_tot)
                    mod_optimizer.gridOptimizer()

                    err_prev = opt_mod_window[model_name]["err_" + model_name]
                    err_new = mod_optimizer.opt_model[model_name]["err_" + model_name]
                    err_delta = err_prev if err_new == 0.0 else (err_prev - err_new)/err_new
                    
                    if err_delta > err_perc_tres:
                        opt_mod_window[model_name] = mod_optimizer.opt_model[model_name].copy()
                        bIter = True
                        window_opt_last = max(window_curr, window_opt_last)
                        curr_time = datetime.now()
                        mod = mod_optimizer.opt_model[model_name]['mod']
                        if (window_opt_last == window_curr): bLastWindowTuned = True
                        if bprint: print("\t (%d, %d) Found TunerWindow model: %s (delta: %.3f, prev: %s new:%s %.3f)" %(window_curr, iIter, model_name, delta_perc, format_number(err_prev), format_number(err_new), err_new/err_prev - 1.0), end="")
                        if bprint: print(", in %s" %(str(curr_time - prev_time).split(".")[0]))
                        if bprint: print("\t\t m:([%.3f, %.3f, %.3f], [%.3f, %.3f, %.3f], %.3f, %.3f, %.3f, %.3f, %d, %d, %d, %d, %.1f, %.1f)" 
                                        %(
                                            min(mod.rg_period), sum(mod.rg_period)/len(mod.rg_period), max(mod.rg_period), 
                                            min(mod.ra_period), sum(mod.ra_period)/len(mod.ra_period), max(mod.ra_period), 
                                            mod.alpha, mod.beta, mod.beta_gcn, mod.gamma, mod.t1, mod.tgi2, mod.tgn2, mod.ta2, mod.Igs_t0, mod.Ias_t0
                                        ))
                        prev_time = curr_time
                    else:
                        #se non trova, prova con il prossimo delta
                        iIterDelta = iIterDelta + 1
                        if iIterDelta < len(deltas):
                            bIter = True
                        # se hai finito i delta, capisci se hai trovato un valore ottimale e rifai ottimizzazione di tutte le window precedenti
                        elif window_curr == window_opt_last and window_curr > 0 and bLastWindowTuned:
                            iIterLastWindowTuned = iIterLastWindowTuned + 1
                            #se non hai mai fatto ottimizzazione window precedenti, falla
                            if iIterLastWindowTuned == 1:
                                window_curr = 0
                                iIterDelta = 0
                                bIter = True
                            else: 
                                iIterDelta = 0
                                iIterLastWindowTuned = 0
                                bLastWindowTuned = False
                                #Mettere qui fine tune
                                opt_mod_window = self.gridAllOptParamFinetuner(opt_mod_window, model_names, finetuned_params = finetuned_params, deltas = [0.1, 0.05, 0.01], grid_steps = 3, opt_max_iter = 100, bprint = True)
                        #se sono al penultimo giro del secondo finetuning, salta l'ultimo perchè sarebbe come rifarlo
                        elif window_curr + window_len == window_opt_last and window_curr > 0 and iIterLastWindowTuned == 1:
                            window_curr = window_curr + window_len
                            iIterDelta = 0
                            iIterLastWindowTuned = 0
                            bLastWindowTuned = False
                            opt_mod_window = self.gridAllOptParamFinetuner(opt_mod_window, model_names, finetuned_params = finetuned_params, deltas = [0.1, 0.05, 0.01], grid_steps = 3, opt_max_iter = 100, bprint = True)
                        else: 
                            iIterDelta = 0
                            #iIterLastWindowTuned = 0
                            #bLastWindowTuned = False

                    iIter = iIter + 1
                    iIterTot = iIterTot + 1
                    
                window_curr = window_curr + window_len

                # adjust final grid to spread last window values until the end of the grid + perform a final optimization based on ovreall grid
            if window_opt_last >-1:
                #grid_param = self.calcGrid(model, param_list = window_params, delta_perc = delta_perc, window_curr = window_curr, window_len = window_len)
                
                rg_period_init = opt_mod_window[model_name]['mod'].rg_period
                ra_period_init = opt_mod_window[model_name]['mod'].ra_period

                if window_opt_last > 0 and self.opt_model_window[model_name]['mod'].rg_period is None:
                    print("********* Window: it is the first time")
                    rg_period = list(map(lambda x, i: x if i in range(0, window_opt_last + window_len) else rg_period_init[window_opt_last], rg_period_init, [j for j in range(0, len(rg_period_init))]))
                    ra_period = list(map(lambda x, i: x if i in range(0, window_opt_last + window_len) else ra_period_init[window_opt_last], ra_period_init, [j for j in range(0, len(ra_period_init))]))

                elif window_opt_last >= 0:
                    iStart = window_curr - 14
                    iEnd = iStart + 14
                    rg_period = list(map(lambda x, i: x if i in range(0, window_curr) else sum(rg_period_init[iStart:iEnd])/len(rg_period_init[iStart:iEnd]), rg_period_init, [j for j in range(0, len(rg_period_init))]))
                    ra_period = list(map(lambda x, i: x if i in range(0, window_curr) else sum(ra_period_init[iStart:iEnd])/len(ra_period_init[iStart:iEnd]), ra_period_init, [j for j in range(0, len(ra_period_init))]))
                    

                param_rg_min = [x*(1 - 0.01) for x in rg_period]
                param_rg_max = [x*(1 + 0.01) for x in rg_period]
                param_ra_min = [x*(1 - 0.01) for x in ra_period]
                param_ra_max = [x*(1 + 0.01) for x in ra_period]

                grid_param.setGridList(Param("rg_period"), [param_rg_min, rg_period, param_rg_max])
                grid_param.setGridList(Param("ra_period"), [param_ra_min, ra_period, param_ra_max])

                mod_optimizer = ModelOptimizer(self.act_data, grid_param, self.model_stepsforward, sync_data_perc = self.sync_data_perc, Pop_tot = self.Pop_tot)
                mod_optimizer.gridOptimizer()
                self.opt_model_window[model_name] = mod_optimizer.opt_model[model_name].copy()

                if bprint: print("\t Saved TunerWindow model: %s (windows: %d, last window: %d, prev: %s new:%s)" %(model_name, window_opt_last/window_len, window_opt_last, format_number(opt_mod_window[model_name]["err_" + model_name]), format_number(mod_optimizer.opt_model[model_name]["err_" + model_name])))
            else:
                print("\t *********** Window: model %s, windows not found ..." %(model_name))

        
        
        if bOptFinetuner: 
            print("****** DOIG FINETUNING IN WINDOW ********") 
            #finetuned_params_init = ['alpha', 'beta', 'beta_gcn', 'gamma', 't1', 'tgi2', 'tgn2', 'ta2', 'Igs_t0', 'Ias_t0']
            #finetuned_params_init = ['rg', 'ra', 'alpha', 'beta', 'gamma']
            finetuned_params_init = ['alpha', 'beta', 'beta_gcn', 'gamma', 't1', 'tgi2', 'tgn2', 'ta2']
            #print("finetuned_params" + str(finetuned_params))
            if finetuned_params is None:
                finetuned_params = finetuned_params_init
            #print("finetuned_params2" + str(finetuned_params))
            self.opt_model_window = self.gridAllOptParamFinetuner(self.opt_model_window, model_names, finetuned_params = finetuned_params, deltas = [0.1, 0.05, 0.01], grid_steps = 3, opt_max_iter = 100, bprint = True)
        #else:
        #    print("****** IN ELSE ********")    



    def gridOptimizer(self):
        initial_time = datetime.now()
        prev_date = initial_time
        self.model_runs = list()
        num_iter = 0
        grid = self.grid

        grid_rg = grid.getGrid('rg')
        grid_ra = grid.getGrid('ra')
        grid_rg_period = grid.getGrid('rg_period')
        if grid_rg_period[0] is not None:
            grid_rg = [grid_rg_period[0][0]]
        
        grid_ra_period = grid.getGrid('ra_period')
        if grid_ra_period[0] is not None:
            grid_ra = [grid_ra_period[0][0]]

        for rg_i in grid_rg:
            for ra_i in grid_ra:
                for alpha_i in grid.getGrid('alpha'):
                    for beta_i in grid.getGrid('beta'):
                        for beta_gcn_i in grid.getGrid('beta_gcn'):
                            for gamma_i in grid.getGrid('gamma'):
                                for t1_i in grid.getGrid('t1', constr_min = 3):
                                    for tgi2_i in grid.getGrid('tgi2', constr_min = t1_i, delta_min = 2):
                                        for tgn2_i in grid.getGrid('tgn2', constr_min = t1_i, delta_min = 2):
                                            for ta2_i in grid.getGrid('ta2', constr_min = 3):
                                                for Igs_t_i in grid.getGrid('Igs_t0'):
                                                    for Igci_t_i in grid.getGrid('Igci_t0'):
                                                        for Igcn_t_i in grid.getGrid('Igcn_t0'):
                                                            for Ias_t_i in grid.getGrid('Ias_t0'):
                                                                for M_t_i in grid.getGrid('M_t0'):
                                                                    for Ggci_t_i in grid.getGrid('Ggci_t0'):
                                                                        for Ggcn_t_i in grid.getGrid('Ggcn_t0'):
                                                                            for Gas_t_i in grid.getGrid('Gas_t0'):
                                                                                for rg_period_i in grid.getGrid('rg_period'):
                                                                                    for ra_period_i in grid.getGrid('ra_period'):

                                                                                        mod = Model(rg=rg_i,
                                                                                                    ra=ra_i,
                                                                                                    alpha=alpha_i,
                                                                                                    beta=beta_i,
                                                                                                    beta_gcn=beta_gcn_i,
                                                                                                    gamma=gamma_i,
                                                                                                    t1=int(t1_i),
                                                                                                    tgi2=int(tgi2_i),
                                                                                                    tgn2=int(tgn2_i),
                                                                                                    ta2=int(ta2_i),
                                                                                                    Igs_t0=int(Igs_t_i),
                                                                                                    Igci_t0=int(Igci_t_i),
                                                                                                    Igcn_t0=int(Igcn_t_i),
                                                                                                    Ias_t0=int(Ias_t_i),
                                                                                                    M_t0=int(M_t_i),
                                                                                                    Ggci_t0=int(Ggci_t_i),
                                                                                                    Ggcn_t0=int(Ggcn_t_i),
                                                                                                    Gas_t0=int(Gas_t_i),
                                                                                                    rg_period=rg_period_i,
                                                                                                    ra_period=ra_period_i,
                                                                                                    Pop_tot=int(self.Pop_tot)
                                                                                                   )

                                                                                        mod.run(self.model_stepsforward)
                                                                                        mod_data = mod_dat(mod)

                                                                                        i_start_act, first_element_act = (self.act_data.i_start, self.act_data.dat_Igc_cum[self.act_data.i_start])
                                                                                        i_start, first_element_mod = find_nearest(mod_data['dat_Igc_cum'], first_element_act)
                                                                                        period = len(self.act_data.dat_Igc[i_start_act:])

                                                                                        err_Igc_cum = np.Inf
                                                                                        err_Igc = np.Inf
                                                                                        err_Gc_cum = np.Inf
                                                                                        err_M_cum = np.Inf
                                                                                        err_Igci_t = np.Inf
                                                                                        err_tot = np.Inf

                                                                                        if ((abs(first_element_mod - first_element_act)/first_element_act)<=self.sync_data_perc):

                                                                                            def errIndex(array1, array2, name = ''):
                                                                                                res = np.Inf
                                                                                                if len(array1) == len(array2):
                                                                                                    #res = np.asarray(list(map(lambda x1, x2: (x1 - x2)/x2, array1, array2)))
                                                                                                    res = np.asarray(list(map(lambda x1, x2: (x1 - x2), array1, array2)))
                                                                                                    res = math.sqrt(np.square(res).mean())
                                                                                                return res
                                                                                    
                                                                                            err_Igc_cum = errIndex(mod_data['dat_Igc_cum'][i_start:i_start + period], self.act_data.dat_Igc_cum[i_start_act:], name='dat_Igc_cum')
                                                                                            err_Igc = errIndex(mod_data['dat_Igc'][i_start:i_start + period], self.act_data.dat_Igc[i_start_act:], name='dat_Igc')
                                                                                            err_Gc_cum = errIndex(mod_data['dat_Gc_cum'][i_start:i_start + period], self.act_data.dat_Gc_cum[i_start_act:], name='dat_Gc_cum')
                                                                                            err_M_cum = errIndex(mod_data['dat_M_cum'][i_start:i_start + period], self.act_data.dat_M_cum[i_start_act:], name='dat_M_cum')
                                                                                            err_Igci_t = errIndex(mod_data['dat_Igci_t'][i_start:i_start + period], self.act_data.dat_Igci_t[i_start_act:], name='dat_Igci_t')
                                                                                            err_tot = (err_Igc_cum + err_Igc + err_Gc_cum + err_M_cum + err_Igci_t)/5

                                                                                        else:
                                                                                            #print("ERROR************k (%d, %d, %d, %d)" %(first_element_act, i_start_act, first_element_mod, i_start))

                                                                                            i_start = np.NaN
                                                                                            err_Igc_cum = np.Inf
                                                                                            err_Igc = np.Inf
                                                                                            err_Gc_cum = np.Inf
                                                                                            err_M_cum = np.Inf
                                                                                            err_Igci_t = np.Inf
                                                                                            err_tot = np.Inf

                                                                                        model_res = {
                                                                                            'mod': copy.copy(mod),
                                                                                            #'mod': copy.deepcopy(mod),
                                                                                            'mod_data': mod_data,
                                                                                            'err_tot': err_tot,
                                                                                            'err_Igc_cum': err_Igc_cum,
                                                                                            'err_Igc': err_Igc,
                                                                                            'err_Gc_cum': err_Gc_cum,
                                                                                            'err_M_cum': err_M_cum,
                                                                                            'err_Igci_t': err_Igci_t,
                                                                                            'i_start': i_start,
                                                                                            'period': period
                                                                                        }

                                                                                        self.model_runs.append([num_iter,
                                                                                                           i_start,
                                                                                                           err_tot,
                                                                                                           err_Igc_cum,
                                                                                                           err_Igc,
                                                                                                           err_Gc_cum,
                                                                                                           err_M_cum,
                                                                                                           period,
                                                                                                           model_res,
                                                                                                           err_Igci_t])


                                                                                        if((num_iter % 100000 == 0) and num_iter>1):
                                                                                            curr_date = datetime.now()
                                                                                            print("iter: %d in %s at %s" %(num_iter, str(curr_date - prev_date), str(curr_date)))
                                                                                            print("\t m:(%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %d, %d, %d, %d)" 
                                                                                              %(rg_i, ra_i, alpha_i, beta_i, beta_gcn_i, gamma_i, t1_i, tgi2_i, tgn2_i, ta2_i), end="")
                                                                                            print(" \t r:(%s, %s, %s, %s, %s)" 
                                                                                              %(format_number(model_res['err_tot']), format_number(model_res['err_Igc_cum']), format_number(model_res['err_Igc']), format_number(model_res['err_Gc_cum']), format_number(model_res['err_M_cum'])))
                                                                                            prev_date = curr_date

                                                                                        if((num_iter % 1000 == 0) and num_iter>1):
                                                                                            self.memorySaver(20)

                                                                                        num_iter = num_iter + 1

        self.memorySaver(20)
        self.optModels(list(self.dic_mod_name.keys()), bprint=False)
        #print("\t tot: iteration: " + str(num_iter))

