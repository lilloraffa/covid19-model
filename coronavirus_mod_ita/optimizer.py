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


class ActualData:
    def __init__(self, data):
        self.date = np.asarray(data['data'])
        self.dat_Igci_t = np.asarray(data['terapia_intensiva'])
        self.dat_Igcn_t = np.asarray(data['ricoverati_con_sintomi']) + np.asarray(data['isolamento_domiciliare'])
        self.dat_Igc = self.dat_Igci_t + self.dat_Igcn_t
        self.dat_Gc_cum = np.asarray(data['dimessi_guariti'])
        self.dat_M_cum = np.asarray(data['deceduti'])
        self.dat_Igc_cum = np.asarray(data['totale_casi'])
        self.i_start = self.getStartTime()
        self.len = len(data['data'])
        self.startDate = data['data'][self.i_start]
        self.endDate = data['data'][self.len-1]
        
    def getStartTime(self):
        start = np.where(self.dat_Igc>5)[0]
        return start[0] if len(start)>0 else None

class Param:
    def __init__(self, par_name, par_min = -1*math.inf, par_max = math.inf):
        self.par_name = par_name
        self.par_min = par_min
        self.par_max = par_max

class GridParam:
    def __init__(self):
        self.param = {}
        self.paramGrid = {}
    
    def setGrid(self, param, grid_avg, grid_min, grid_max, steps):
        if (param.par_max is not None) and (param.par_min is not None):
            val_max = grid_max if grid_max <= param.par_max else param.par_max
            val_min = grid_min if grid_min >= param.par_min else param.par_min
        else:
            val_max = grid_max
            val_min = grid_min
        self.paramGrid[param.par_name] = self.getList(val_min, val_max, grid_avg, steps)
        self.param[param.par_name] = param
    
    def setGridList(self, param, grid_list):
        self.paramGrid[param.par_name] = grid_list
        self.param[param.par_name] = param
        
    def getList(self, val_min, val_max, val_avg, steps):
        mod = (steps - 1) % 2
        steps_half = (steps-1)/2
        gridList = []
        steps_min = math.floor(steps_half)
        steps_max = math.floor(steps_half)
        if(mod > 0):
            if((val_max-val_avg) > (val_avg - val_min)):
                steps_min = math.floor(steps_half) + 1 
            else:
                steps_max = math.floor(steps_half) + 1

        if steps > 2:
            gridList = np.arange(val_min, val_avg, (val_avg - val_min)/(steps_min)).tolist()
            gridList = gridList + np.arange(val_avg, val_max, (val_max - val_avg)/(steps_max)).tolist()
            gridList.append(val_max)
        elif steps == 2:
            gridList = [val_min, val_max]
        else:
            gridList = [val_avg]
        return gridList
        
    
    def getGrid(self, par_name, constr_min = None, constr_max = None, delta_min=0):
        if par_name in self.paramGrid.keys():
            grid = self.paramGrid[par_name]

            if grid[0] is not None:
                if isinstance(grid[0], list):
                    #for i in range(0,len(grid)):
                    #    grid[i] = [ x for x in grid[i] if (x >= self.param[par_name].par_min and x <= self.param[par_name].par_max)]
                    #res = 
                    res = []

                    for elem in grid:
                        if constr_min is not None:
                            if(elem[0]< constr_min):
                                delta = constr_min - elem[0] + delta_min
                                elem = [x + delta for x in elem]
                        if constr_max is not None:
                            if(elem[len(elem)-1] > constr_max):
                                elem = [x for x in elem if x<= constr_max]
                        res.append([ x for x in elem if (x >= self.param[par_name].par_min and x <= self.param[par_name].par_max)])

                    return res
                else:
                    if constr_min is not None:
                        if(grid[0]< constr_min):
                            delta = constr_min - grid[0] + delta_min
                            grid = [x + delta for x in grid]
                    if constr_max is not None:
                        if(grid[len(grid)-1] > constr_max):
                            grid = [x for x in grid if x<= constr_max]
                    return [ x for x in grid if (x >= self.param[par_name].par_min and x <= self.param[par_name].par_max)]
            else:
                return [None]
        else:
            return [None]

class ModelOptimizer:
    dic_mod_name = {
        'tot': 2,
        'Igc_cum': 3,
        'Igc': 4,
        'Gc_cum': 5,
        'M_cum': 6
        #'Igci_t': 9
    }
    def __init__(self, act_data, grid, model_stepsforward, sync_data_perc = 0.1, window_len = 7, Pop_tot = 50000000):
        self.act_data = act_data
        self.grid = grid
        self.sync_data_perc = sync_data_perc
        self.model_stepsforward = model_stepsforward
        self.model_runs = list()
        self.opt_model = {}
        self.opt_model_initial = None
        self.opt_model_window = None
        self.window_len = window_len
        self.Pop_tot = Pop_tot
        
    def start(self):
        initial_time = datetime.now()
        print("**** Model optimization started at: " + str(initial_time))
        self.gridOptimizer()
        final_time = datetime.now()
        print("**** Model optimization ended at: " + str(final_time))
        print("\t in: " + str(final_time - initial_time))
        
        initial_time_new = final_time
        print("**** Model AllOptParamFinetuner started at: " + str(initial_time_new))
        model_names = ['tot', 'Igc_cum', 'Igc', 'Gc_cum', 'M_cum']
        self.opt_model = self.gridAllOptParamFinetuner(self.opt_model, model_names, delta_perc = 0.1, grid_steps = 2, opt_max_iter = 100, bprint = True)
        self.opt_model = self.gridAllOptParamFinetuner(self.opt_model, model_names, delta_perc = 0.05, grid_steps = 2, opt_max_iter = 100, bprint = True)
        self.opt_model = self.gridAllOptParamFinetuner(self.opt_model, model_names, delta_perc = 0.01, grid_steps = 2, opt_max_iter = 100, bprint = True)
        
        final_time = datetime.now()
        print("**** Model AllOptParamFinetuner ended at: " + str(final_time))
        print("\t in: " + str(final_time - initial_time_new))
        
        initial_time_new = final_time
        print("**** Model WindowOptimizer started at: " + str(initial_time_new))
        initial_time_new = final_time
        model_names = ['tot', 'Igc_cum', 'Igc', 'Gc_cum', 'M_cum']
        self.windowParamOptimizer(model_names, self.window_len, delta_perc = 0.1, opt_max_iter = 100, bprint=True)
        #self.windowParamOptimizer(model_names, self.window_len, delta_perc = 0.05, opt_max_iter = 100, bprint=True)
        self.windowParamOptimizer(model_names, self.window_len, delta_perc = 0.01, opt_max_iter = 100, bOptFinetuner = False, bprint=True)
        
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
            print("Opt. model: %s \t (%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %d, %d, %d, %d) \n\t err_tot= %.2f \t err_Igc_cum= %.2f \t err_G= %.2f \t err_M= %.2f" %(
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
                model['err_tot'],
                model['err_Igc_cum'],
                model['err_Gc_cum'],
                model['err_M_cum']
            ))
        
    
    def optModelSel(self, mod_name, bprint=False):
        #print("mod: %s - %s" %(mod_name, str(np.asarray(self.model_runs)[:,self.dic_mod_name[mod_name]].astype(float))))
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

    
    def gridAllOptParamFinetuner(self, optmodels_in, model_names, finetuned_params = None, delta_perc = 0.025, grid_steps = 2, opt_max_iter = 100, bprint = False):
        initial_time = datetime.now()
        #if self.opt_model_initial is None:
        #    self.opt_model_initial = self.opt_model.copy()
        
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
            #finetuned_params_final = ['rg', 'ra', 'alpha', 'beta', 'gamma', 't1', 'tgi2', 'tgn2']
            finetuned_params_final = ['rg', 'ra', 'alpha', 'beta', 'beta_gcn', 'gamma', 't1', 'tgi2', 'tgn2', 'ta2', 'Igs_t0', 'Ias_t0']
            #finetuned_params_final = ['rg', 'ra', 'alpha', 'beta', 'beta_gcn', 'gamma']
            
        for param in finetuned_params_final:
                params[param] = grid_steps
                
        
        for model_name in model_names:
            
            bIter = True
            iIter = 0
            
            while bIter and iIter < opt_max_iter :
                bIter = False
                iIter = iIter + 1
                grid_param = GridParam()
                model = optmodels[model_name]['mod']
                
                for param in params.keys():
                    if param in ['t1', 'tgi2', 'tgn2', 'ta2']:
                        delta_inc = 1
                        if delta_perc == 0.1:
                            delta_inc = 2
                        param_val = getattr(model, param)
                        param_min = 2 if param_val <= 2 else max(param_val - delta_inc, 2)
                        grid_param.setGrid(Param(param, par_min = 2, par_max=40), grid_avg = param_val, grid_min = param_min, grid_max = min(param_val + delta_inc, 2), steps = params[param])
                    if param in ['rg_period', 'ra_period']:
                        param_value = getattr(model, param)
                        grid_param.setGridList(Param(param), [param_value])
                    else: 
                        param_val = getattr(model, param)
                        grid_param.setGrid(Param(param), grid_avg = param_val, grid_min = (param_val*(1-delta_perc)), grid_max = (param_val*(1+delta_perc)), steps = params[param])

                #Find the optimal models with new grid
                mod_optimizer = ModelOptimizer(self.act_data, grid_param, self.model_stepsforward, sync_data_perc = self.sync_data_perc, Pop_tot = self.Pop_tot)
                mod_optimizer.gridOptimizer()

                err_prev = optmodels[model_name]["err_" + model_name]
                err_new = mod_optimizer.opt_model[model_name]["err_" + model_name]
                err_delta = err_prev if err_new == 0.0 else (err_prev - err_new)/err_new
                #print("\t while (%d, %s, %f, %f, %f, %f.2f)" %(iIter, model_name, delta_perc, err_prev, err_new, err_new/err_prev - 1.0) )
                #print("\t\t" + str(grid_param.paramGrid))
                #print("\t while (%d, %s, error_prev=%f, error_new=%f)" %(iIter, model_name, err_prev, err_new) )
                if err_delta > 0.001:
                    optmodels[model_name] = mod_optimizer.opt_model[model_name].copy()
                    bIter = True
                    if bprint: print("\t (%d) Found FineTuner better model: %s (delta: %f, prev: %f -  new:%f - %.2f)" %(iIter, model_name, delta_perc, err_prev, err_new, err_new/err_prev - 1.0))
                    if bprint: print("\t\t in %s" %(str(datetime.now() - initial_time)))
        
        return optmodels
    
    def windowParamOptimizer(self, model_names, window_len, delta_perc = 0.1, opt_max_iter = 100, bOptFinetuner = True, bprint = False):
        if self.opt_model_window is None:
            opt_mod_window = self.opt_model.copy()
            self.opt_model_window = self.opt_model.copy()
        else:
            opt_mod_window = self.opt_model_window.copy()
        
        for model_name in model_names:
            
            #model = opt_mod_window[model_name]['mod']
            mod_istart = opt_mod_window[model_name]['i_start']
            window_curr = 0
            window_opt_last = None
            grid_param_opt = None
            while (window_curr - mod_istart) <= self.act_data.len   :
                #print("Win: (%d, %d, %d) - data_uff_len: %d" %(window_curr, window_curr + window_len, mod_istart, self.act_data.len))
                bIter = True
                iIter = 0
                while bIter and iIter < opt_max_iter :
                    model = opt_mod_window[model_name]['mod']
                    bIter = False
                    #iIter = iIter + 1
                    grid_param = GridParam()
                    
                    for param in model.params.keys():
                        param_value = getattr(model, param)
                        if param in ['rg_period', 'ra_period']:
                            if param_value is None:
                                val = model.params['rg'] if param == 'rg_period' else model.params['ra']
                                tot_len = len(opt_mod_window[model_name]['mod_data']['dat_Igc'])
                                param_value = [val for i in range(0, tot_len)]
                                #print("if None period: ", param, val, tot_len)
                            
                            if iIter == 0 and window_curr>0:
                                # At the first iteration for the window, set the starting window equals to the previous value
                                #print(param_value[window_curr - 1], param_value[window_curr])
                                #print(param + " before - curr: %d" %(window_curr) + " - " + str(param_value))
                                
                                #param_value = list(map(lambda x, i: param_value[window_curr - 1] if i in range(window_curr, len(param_value)) else x, param_value, [j for j in range(0, len(param_value))]))
                                
                                param_min1 = list(map(lambda x, i: param_value[window_curr - 1]*(1 - delta_perc) if i in range(window_curr, window_curr + window_len) else x, param_value, [j for j in range(0, len(param_value))]))
                                param_max1 = list(map(lambda x, i: param_value[window_curr - 1]*(1 + delta_perc) if i in range(window_curr, window_curr + window_len) else x, param_value, [j for j in range(0, len(param_value))]))
                                #print(param + "after - curr: %d" %(window_curr) + " - " + str(param_min1[0:60]))
                            else:
                                #print(param + " test: %d - %d" %(window_curr, iIter) + " - " + str(param_value))
                                param_min1 = list(map(lambda x, i: x*(1 - delta_perc) if i in range(window_curr, window_curr + window_len) else x, param_value, [j for j in range(0, len(param_value))]))
                                param_max1 = list(map(lambda x, i: x*(1 + delta_perc) if i in range(window_curr, window_curr + window_len) else x, param_value, [j for j in range(0, len(param_value))]))
                                #print(param + " test: %d - %d" %(window_curr, iIter) + " - " + str(param_min1[0:60]))
                            # param_min = list(map(lambda x, i: x*(1 - delta_perc) if i in range(window_curr, window_curr + window_len) else x, param_value, [j for j in range(0, len(param_value))]))
                            # param_max = list(map(lambda x, i: x*(1 + delta_perc) if i in range(window_curr, window_curr + window_len) else x, param_value, [j for j in range(0, len(param_value))]))
                            param_grid = [param_min1, param_max1]
                            grid_param.setGridList(Param(param), param_grid)
                        else: 
                            #print(param, param_value)
                            grid_param.setGrid(Param(param), grid_avg = param_value, grid_min = param_value, grid_max = param_value, steps = 1)
                    #print(grid_param.getGrid('rg_period')[0])
                        
                    #Find the optimal models with new grid
                    mod_optimizer = ModelOptimizer(self.act_data, grid_param, self.model_stepsforward, sync_data_perc = self.sync_data_perc, Pop_tot = self.Pop_tot)
                    mod_optimizer.gridOptimizer()

                    err_prev = opt_mod_window[model_name]["err_" + model_name]
                    err_new = mod_optimizer.opt_model[model_name]["err_" + model_name]
                    err_delta = err_prev if err_new == 0.0 else (err_prev - err_new)/err_new
                    #print("\t while (%d, %d, %s, error_prev=%f, error_new=%f, %.5f, %.5f)" %(iIter, window_curr, model_name, err_prev, err_new, err_new/err_prev - 1.0, ((err_prev - err_new)/err_new)) )
                    #print("\t while (%d, %d, %s, error_prev=%f, error_new=%f)" %(iIter, window_curr, model_name, err_prev, err_new) )
                    if err_delta > 0.001:
                        opt_mod_window[model_name] = mod_optimizer.opt_model[model_name].copy()
                        grid_param_opt = grid_param
                        bIter = True
                        window_opt_last = window_curr
                        if bprint: print("\t (%d, %d) Found TunerWindow model: %s (delta: %f, prev: %f new:%f %.2f)" %(window_curr, iIter, model_name, delta_perc, err_prev, err_new, err_new/err_prev - 1.0))
                    iIter = iIter + 1
                window_curr = window_curr + window_len

            # adjust final grid to spread last window values until the end of the grid + perform a final optimization based on ovreall grid
            if window_opt_last is not None:
                rg_period_init = opt_mod_window[model_name]['mod'].rg_period
                ra_period_init = opt_mod_window[model_name]['mod'].ra_period
                #print("Final Window: %s -  %d" %(model_name, window_opt_last))
                #print("beginning rg_period: " + str(rg_period))
                #print()
                rg_period = list(map(lambda x, i: x if i in range(0, window_opt_last + window_len) else rg_period_init[window_opt_last], rg_period_init, [j for j in range(0, len(rg_period_init))]))
                ra_period = list(map(lambda x, i: x if i in range(0, window_opt_last + window_len) else ra_period_init[window_opt_last], ra_period_init, [j for j in range(0, len(ra_period_init))]))
                #print ("after rg_period: " + str(rg_period))
                #print()
                param_rg_min = [x*(1 - 0.01) for x in rg_period]
                param_rg_max = [x*(1 + 0.01) for x in rg_period]
                param_ra_min = [x*(1 - 0.01) for x in ra_period]
                param_ra_max = [x*(1 + 0.01) for x in ra_period]

                grid_param.setGridList(Param("rg_period"), [param_rg_min, rg_period, rg_period_init, param_rg_max])
                grid_param.setGridList(Param("ra_period"), [param_ra_min, ra_period, ra_period_init, param_ra_max])
                #print(param_rg_min)
                #print(param_rg_max)
                mod_optimizer = ModelOptimizer(self.act_data, grid_param, self.model_stepsforward, sync_data_perc = self.sync_data_perc, Pop_tot = self.Pop_tot)
                mod_optimizer.gridOptimizer()

                err_prev = opt_mod_window[model_name]["err_" + model_name]
                err_new = mod_optimizer.opt_model[model_name]["err_" + model_name]
                err_delta = err_prev if err_new == 0.0 else (err_prev - err_new)/err_new
                #print("\t while (%d, %d, %s, error_prev=%f, error_new=%f, %.5f, %.5f)" %(iIter, window_curr, model_name, err_prev, err_new, err_new/err_prev - 1.0, ((err_prev - err_new)/err_new)) )
                print("\t Windows last saving (%d, %d, %s, error_prev=%f, error_new=%f)" %(iIter, window_opt_last, model_name, err_prev, err_new) )
                if err_delta > 0:
                    self.opt_model_window[model_name] = mod_optimizer.opt_model[model_name].copy()
                    print("\t It is optimal, save the new one")
                    #print(mod_optimizer.opt_model[model_name]['mod'].rg_period)
                else:
                    self.opt_model_window[model_name] = opt_mod_window[model_name].copy()
                    print("\t It is not optimal, save the old one")
                    #print(opt_mod_window[model_name]['mod'].rg_period)
                
                if bprint: print("\t Saved TunerWindow model: %s (windows: %d, last window: %d, prev: %f new:%f)" %(model_name, window_opt_last/window_len, window_opt_last, opt_mod_window[model_name]["err_" + model_name], mod_optimizer.opt_model[model_name]["err_" + model_name]))
            else:
                print("model %s not entered in last mile..." %(model_name))
                #print(opt_mod_window[model_name]['mod'].rg_period)

        finetuned_params = ['alpha', 'beta', 'beta_gcn', 'gamma', 't1', 'tgi2', 'tgn2', 'ta2', 'Igs_t0', 'Ias_t0']
        #finetuned_params = ['rg', 'ra', 'alpha', 'beta', 'gamma', 't1', 'tgi2', 'tgn2']
        if bOptFinetuner: self.opt_model_window = self.gridAllOptParamFinetuner(self.opt_model_window, model_names, delta_perc = 0.01, grid_steps = 2, opt_max_iter = 100, finetuned_params = finetuned_params, bprint = True)       
    


    def gridOptimizer(self):
        initial_time = datetime.now()
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
                                for t1_i in grid.getGrid('t1'):
                                    for tgi2_i in grid.getGrid('tgi2', constr_min = t1_i, delta_min = 2):
                                        for tgn2_i in grid.getGrid('tgn2', constr_min = t1_i, delta_min = 2):
                                            for ta2_i in grid.getGrid('ta2'):
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
                                                                                        now = datetime.now()
                                                                                        #print("**model:(%.3f, %.3f, %.3f, %.3f, %.3f, %d, %d, %d, %d)" 
                                                                                        #      %(rg_i, ra_i, alpha_i, beta_i, gamma_i, t1_i, tgi2_i, tgn2_i, ta2_i))
                                                                                              #end="")

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

                                                                                        i_start_act, first_element_act = (self.act_data.i_start, self.act_data.dat_Igc[self.act_data.i_start])
                                                                                        i_start, first_element_mod = find_nearest(mod_data['dat_Igc'], first_element_act)
                                                                                        period = len(self.act_data.dat_Igc[i_start_act:])

                                                                                        err_Igc_cum = np.Inf
                                                                                        err_Igc = np.Inf
                                                                                        err_Gc_cum = np.Inf
                                                                                        err_M_cum = np.Inf
                                                                                        err_Igci_t = np.Inf
                                                                                        err_tot = np.Inf

                                                                                        if ((abs(first_element_mod - first_element_act)/first_element_act)<=self.sync_data_perc):


                                                                                            #print("START** \t (%d, %d, %d, %d)" %(i_start, period, len(mod_data['dat_Igc_cum']), len(mod_data['dat_Igc_cum'][i_start:i_start + period])))

                                                                                            def errIndex(array1, array2, name = ''):
                                                                                                res = np.Inf
                                                                                                #print("errIndex:", len(array1), len(array2), str(array1), str(array2))
                                                                                                if len(array1) == len(array2):
                                                                                                    #print("errIndex:", str(array1), str(array2), str(list(map(lambda x1, x2: x1 if x2==0.0 else (x1 - x2)/x2, array1, array2))))
                                                                                                    #res = np.asarray(list(map(lambda x1, x2: x1 if x2==0.0 else (x1 - x2)/x2, array1, array2)))
                                                                                                    #res = np.asarray(list(map(lambda x1, x2: (x1 - x2)/x2, array1, array2)))
                                                                                                    #error ver x.abs
                                                                                                    res = np.asarray(list(map(lambda x1, x2: (x1 - x2), array1, array2)))
                                                                                                    res = np.square(res).mean()
                                                                                                return res
                                                                                            
                                                                                            
                                                                                            err_Igc_cum = errIndex(mod_data['dat_Igc_cum'][i_start:i_start + period], self.act_data.dat_Igc_cum[i_start_act:], name='dat_Igc_cum')
                                                                                            err_Igc = errIndex(mod_data['dat_Igc'][i_start:i_start + period], self.act_data.dat_Igc[i_start_act:], name='dat_Igc')
                                                                                            err_Gc_cum = errIndex(mod_data['dat_Gc_cum'][i_start:i_start + period], self.act_data.dat_Gc_cum[i_start_act:], name='dat_Gc_cum')
                                                                                            err_M_cum = errIndex(mod_data['dat_M_cum'][i_start:i_start + period], self.act_data.dat_M_cum[i_start_act:], name='dat_M_cum')
                                                                                            err_Igci_t = errIndex(mod_data['dat_Igci_t'][i_start:i_start + period], self.act_data.dat_Igci_t[i_start_act:], name='dat_Igci_t')
                                                                                            
                                                                                            #Error ver 1
                                                                                            err_tot = (err_Igc_cum + err_Igc + err_Gc_cum + err_M_cum + err_Igci_t)/5
                                                                                            
                                                                                            #Error ver 2
                                                                                            #err_tot = ((math.sqrt(err_Igc_cum) + math.sqrt(err_Igc) + math.sqrt(err_Gc_cum) + math.sqrt(err_M_cum))/4) ** 2
                                                                                            
                                                                                            #Error ver 3
                                                                                            #err_tot = ((math.sqrt(err_Igc_cum) + math.sqrt(err_Igc) + 0.6*math.sqrt(err_Gc_cum) + 0.6*math.sqrt(err_M_cum))/3.2) ** 2

                                                                                            #Error ver 4
                                                                                            #err_tot = ((math.sqrt(err_Igc_cum) + math.sqrt(err_Igc) + math.sqrt(err_Gc_cum) + math.sqrt(err_M_cum))/4)
                                                                                            
                                                                                            
                                                                                            #old - can be canceled maybe
                                                                                            # err_Igc_cum = errIndex(mod_data['dat_Igc_cum'][i_start:i_start + period], self.act_data.dat_Igc_cum, name='dat_Igc_cum')
                                                                                            # err_Igc = errIndex(mod_data['dat_Igc'][i_start:i_start + period], self.act_data.dat_Igc, name='dat_Igc')
                                                                                            # err_Gc_cum = errIndex(mod_data['dat_Gc_cum'][i_start:i_start + period], self.act_data.dat_Gc_cum, name='dat_Gc_cum')
                                                                                            # err_M_cum = errIndex(mod_data['dat_M_cum'][i_start:i_start + period], self.act_data.dat_M_cum, name='dat_M_cum')
                                                                                            # err_Igci_t = errIndex(mod_data['dat_Igci_t'][i_start:i_start + period], self.act_data.dat_Igci_t, name='dat_Igci_t')
                                                                                            #(err_tot, err_Igc_cum, err_Igc, err_Gc_cum, err_M_cum, err_Igci_t) = 

                                                                                            #print("********ERRORS1: ", err_tot, err_Igc_cum, err_Igc, err_Gc_cum, err_M_cum, err_Igci_t)
                                                                                            
                                                                                            # Error ver x.pen
                                                                                            #tot_penality = math.sqrt(err_tot)
                                                                                            tot_penality = err_tot
                                                                                            err_Igc_cum = 0.5*err_Igc_cum + 0.5*tot_penality
                                                                                            err_Igc = 0.5*err_Igc + 0.5*tot_penality
                                                                                            err_Gc_cum = 0.5*err_Gc_cum + 0.5*tot_penality
                                                                                            err_M_cum = 0.5*err_M_cum + 0.5*tot_penality
                                                                                            err_Igci_t = 0.5*err_Igci_t + 0.5*tot_penality
                                                                                            #print("********ERRORS2: ", err_tot, err_Igc_cum, err_Igc, err_Gc_cum, err_M_cum, err_Igci_t)

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
                                                                                            'mod': copy.deepcopy(mod),
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

                                                                                        #print(" \t result:(%.3f, %.3f, %.3f, %.3f, %.3f)" 
                                                                                        #     %(model_res['err_tot'], model_res['err_Igc_cum'], model_res['err_Igc'], model_res['err_Gc_cum'], model_res['err_M_cum']))

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


                                                                                        if((num_iter % 50000 == 0) and num_iter>1):
                                                                                            print("iter: %d in %s at %s" %(num_iter, str(datetime.now() - initial_time), str(datetime.now())))
                                                                                            print("\t m:(%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %d, %d, %d, %d)" 
                                                                                              %(rg_i, ra_i, alpha_i, beta_i, beta_i, gamma_i, t1_i, tgi2_i, tgn2_i, ta2_i), end="")
                                                                                            print(" \t r:(%.3f, %.3f, %.3f, %.3f, %.3f)" 
                                                                                              %(model_res['err_tot'], model_res['err_Igc_cum'], model_res['err_Igc'], model_res['err_Gc_cum'], model_res['err_M_cum']))


                                                                                        if((num_iter % 1000 == 0) and num_iter>1):
                                                                                            self.memorySaver(20)

                                                                                        num_iter = num_iter + 1

        self.memorySaver(20)
        self.optModels(list(self.dic_mod_name.keys()), bprint=False)
        #print("\t tot: iteration: " + str(num_iter))

