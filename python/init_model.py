from epidemic_model import *
from data_mgmt import *
import pandas as pd
import time


def getData(geo = 'ita', mod_name = '', obj_path = '', rel_path=''):
    """
    (ImportData, Object)
    """
    if geo == 'ita':
        return ImportData("data/covid_ita_regional.csv", 
                            ext_csv_anagr = "data/anagr_ita_regional.csv", 
                            rel_path = rel_path
                            ), importModel(rel_path + obj_path + mod_name)



def importModel(path):
    opt_mod_obj = {'model': {}}
    if os.path.exists(path):
        with open(path, 'rb') as f:
            try:
                opt_mod_obj = pickle.load(f)
            except:
                print("error")
    return opt_mod_obj

def saveOptMod(obj, actual_data_update, path = "", mod_name="generic", bDataUpdateTime = True, bRunDatetime = True):
    tempFile = path+"__model-saving"
    while os.path.isfile(tempFile + mod_name):
        time.sleep(5)
    else:
        open(tempFile, 'a').close()
    
    path_full = path + mod_name
    obj_mod = importModel(path_full)

    for model in obj['model'].keys():
        obj_mod['model'][model] = obj['model'][model]

    #data_uff.date[len(data_uff.date)-1]
    actual_data_update = np.datetime_as_string(actual_data_update, unit='D')
    current_date = np.datetime_as_string(np.datetime64(datetime.now()), unit='s')
    
    if bDataUpdateTime: path_full = path_full + "_act-" + actual_data_update
    if bRunDatetime: path_full = path_full + "_" + current_date 
    with open(path_full, 'wb') as saved_file:
        pickle.dump(obj_mod, saved_file)
    if os.path.exists(tempFile):
        os.remove(tempFile)

def saveOptMod2(obj, actual_data_update, path = "", mod_name="generic", bDataUpdateTime = True, bRunDatetime = True):
    
    #data_uff.date[len(data_uff.date)-1]
    actual_data_update = np.datetime_as_string(actual_data_update, unit='D')
    current_date = np.datetime_as_string(np.datetime64(datetime.now()), unit='s')
    path_full = path + mod_name
    if bDataUpdateTime: path_full = path_full + "_act-" + actual_data_update
    if bRunDatetime: path_full = path_full + "_" + current_date 
    with open(path_full, 'wb') as saved_file:
        pickle.dump(obj, saved_file)



def get_grid_param(params, delta_mult = 3, steps = 3, exclude = None):
    param_init, exclude_param = getParamList(exclude = exclude)
    mod_generic = Model()
    grid_param = GridParam()

    for param in mod_generic.params.keys():
        if param in exclude_param:
            param_avg = params[param]
            grid_param.setGrid(Param(param), grid_avg = param_avg, grid_min = param_avg, grid_max = param_avg, steps = 1)
        elif param in ['t1', 'tgi2', 'tgn2', 'ta2', 'Igs_t0', 'Ias_t0']:
            par_min = 1 if param in ['t1', 'tgi2', 'tgn2', 'ta2'] else 0
            param_avg = params[param]
            #param_min = param_avg - ((delta_mult - 1)/delta_mult)*(param_avg - 1)
            #param_max = param_avg * delta_mult
            param_min = round(param_avg/delta_mult, 0)
            param_max = param_avg*delta_mult
            if param_min == param_max:
                param_min = max(param_avg - 1, 0)
                param_avg = max(param_avg, 1)
                param_max = param_avg + 1


            grid_param.setGrid(Param(param, par_min = par_min), grid_avg = param_avg, grid_min = param_min, grid_max = param_max, steps = steps)
        elif param in ['rg_period', 'ra_period']:
            grid_param.setGridList(Param(param), [params[param]])
        else: 
            param_avg = params[param]
            param_min = param_avg - ((delta_mult - 1)/delta_mult)*(param_avg)
            param_max = param_avg * delta_mult
            grid_param.setGrid(Param(param), grid_avg = param_avg, grid_min = param_min, grid_max = param_max, steps = steps)
    #print(grid_param.paramGrid)
    return grid_param

def load_grid(grid_name):
    grid_param = GridParam()
    if grid_name == 'ita-aggr':
        grid_param.setGrid(Param('rg', par_min = 0.000001), grid_avg = 0.2, grid_min = 0.13, grid_max = 0.35, steps = 3)
        grid_param.setGrid(Param('ra', par_min = 0.000001), grid_avg = 0.731904, grid_min = 0.5, grid_max = 0.9, steps = 3)
        grid_param.setGrid(Param('alpha', par_min = 0.00000001), grid_avg = 0.605759, grid_min = 0.5, grid_max = 0.75, steps = 3)
        grid_param.setGrid(Param('beta', par_min = 0.00000001), grid_avg = 0.119766, grid_min = 0.01, grid_max = 0.2, steps = 3)
        grid_param.setGrid(Param('beta_gcn', par_min = 0.00000001), grid_avg = 0.007077, grid_min = 0.003, grid_max = 0.02, steps = 3)
        grid_param.setGrid(Param('gamma', par_min = 0.00000001), grid_avg = 0.09096, grid_min = 0.05, grid_max = 0.15, steps = 3)
        grid_param.setGrid(Param('t1', par_min = 1), grid_avg = 4, grid_min = 4, grid_max = 6, steps = 2)
        grid_param.setGrid(Param('tgi2', par_min = 1), grid_avg = 23, grid_min = 15, grid_max = 25, steps = 2)
        grid_param.setGrid(Param('tgn2', par_min = 1), grid_avg = 18, grid_min = 15, grid_max = 25, steps = 2)
        grid_param.setGrid(Param('ta2', par_min = 1), grid_avg = 6, grid_min = 4, grid_max = 6, steps = 2)
        grid_param.setGrid(Param('Igs_t0', par_min = 0), grid_avg = 221, grid_min = 20, grid_max = 100, steps = 2)
        grid_param.setGrid(Param('Igci_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 100, steps = 1)
        grid_param.setGrid(Param('Igcn_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ias_t0', par_min = 0), grid_avg = 210, grid_min = 20, grid_max = 100, steps = 2)
        grid_param.setGrid(Param('M_t0', par_min = 0), grid_avg = 0, grid_min = 70, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ggci_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ggcn_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Gas_t0', par_min = 0), grid_avg = 0, grid_min = 100, grid_max = 500, steps = 1)
    if grid_name == 'ita-aggr_old':
        grid_param.setGrid(Param('rg', par_min = 0.000001), grid_avg = 0.2, grid_min = 0.13, grid_max = 0.35, steps = 4)
        grid_param.setGrid(Param('ra', par_min = 0.000001), grid_avg = 0.731904, grid_min = 0.5, grid_max = 0.9, steps = 4)
        grid_param.setGrid(Param('alpha', par_min = 0.00000001), grid_avg = 0.605759, grid_min = 0.5, grid_max = 0.75, steps = 4)
        grid_param.setGrid(Param('beta', par_min = 0.00000001), grid_avg = 0.119766, grid_min = 0.01, grid_max = 0.2, steps = 4)
        grid_param.setGrid(Param('beta_gcn', par_min = 0.00000001), grid_avg = 0.007077, grid_min = 0.003, grid_max = 0.02, steps = 3)
        grid_param.setGrid(Param('gamma', par_min = 0.00000001), grid_avg = 0.09096, grid_min = 0.05, grid_max = 0.15, steps = 3)
        grid_param.setGrid(Param('t1', par_min = 1), grid_avg = 4, grid_min = 3, grid_max = 6, steps = 3)
        grid_param.setGrid(Param('tgi2', par_min = 1), grid_avg = 23, grid_min = 18, grid_max = 25, steps = 3)
        grid_param.setGrid(Param('tgn2', par_min = 1), grid_avg = 18, grid_min = 14, grid_max = 25, steps = 3)
        grid_param.setGrid(Param('ta2', par_min = 1), grid_avg = 6, grid_min = 3, grid_max = 15, steps = 3)
        grid_param.setGrid(Param('Igs_t0', par_min = 0), grid_avg = 221, grid_min = 50, grid_max = 300, steps = 3)
        grid_param.setGrid(Param('Igci_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Igcn_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ias_t0', par_min = 0), grid_avg = 210, grid_min = 50, grid_max = 300, steps = 3)
        grid_param.setGrid(Param('M_t0', par_min = 0), grid_avg = 0, grid_min = 70, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ggci_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ggcn_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Gas_t0', par_min = 0), grid_avg = 0, grid_min = 100, grid_max = 500, steps = 1)
    if grid_name == 'ita-aggr_fixed':
        grid_param.setGrid(Param('rg', par_min = 0.000001), grid_avg = 0.18421, grid_min = 0.10, grid_max = 0.35, steps = 3)
        grid_param.setGrid(Param('ra', par_min = 0.000001), grid_avg = 0.752465, grid_min = 0.5, grid_max = 0.9, steps = 3)
        grid_param.setGrid(Param('alpha', par_min = 0.00000001), grid_avg = 0.868725, grid_min = 0.6, grid_max = 0.95, steps = 3)
        grid_param.setGrid(Param('beta', par_min = 0.00000001), grid_avg = 0.083776, grid_min = 0.007, grid_max = 0.2, steps = 3)
        grid_param.setGrid(Param('beta_gcn', par_min = 0.00000001), grid_avg = 0.083776, grid_min = 0.007, grid_max = 0.2, steps = 3)
        grid_param.setGrid(Param('gamma', par_min = 0.00000001), grid_avg = 0.223425, grid_min = 0.1, grid_max = 0.40, steps = 3)
        grid_param.setGrid(Param('t1', par_min = 1), grid_avg = 4, grid_min = 4, grid_max = 6, steps = 2)
        grid_param.setGrid(Param('tgi2', par_min = 1), grid_avg = 12, grid_min = 15, grid_max = 25, steps = 2)
        grid_param.setGrid(Param('tgn2', par_min = 1), grid_avg = 20, grid_min = 15, grid_max = 25, steps = 2)
        grid_param.setGrid(Param('ta2', par_min = 1), grid_avg = 4, grid_min = 4, grid_max = 6, steps = 2)
        grid_param.setGrid(Param('Igs_t0', par_min = 0), grid_avg = 252, grid_min = 10, grid_max = 50, steps = 2)
        grid_param.setGrid(Param('Igci_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Igcn_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ias_t0', par_min = 0), grid_avg = 216, grid_min = 10, grid_max = 50, steps = 2)
        grid_param.setGrid(Param('M_t0', par_min = 0), grid_avg = 0, grid_min = 70, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ggci_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ggcn_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Gas_t0', par_min = 0), grid_avg = 0, grid_min = 100, grid_max = 500, steps = 1)
    elif grid_name == 'ita-aggr_fast':
        grid_param.setGrid(Param('rg', par_min = 0.000001), grid_avg = 0.2, grid_min = 0.07, grid_max = 0.15, steps = 2)
        grid_param.setGrid(Param('ra', par_min = 0.000001), grid_avg = 0.731904, grid_min = 0.1063, grid_max = 0.9, steps = 2)
        grid_param.setGrid(Param('alpha', par_min = 0.00000001), grid_avg = 0.605759, grid_min = 0.807, grid_max = 0.85, steps = 2)
        grid_param.setGrid(Param('beta', par_min = 0.00000001), grid_avg = 0.119766, grid_min = 0.0114, grid_max = 0.2, steps = 2)
        grid_param.setGrid(Param('beta_gcn', par_min = 0.00000001), grid_avg = 0.007077, grid_min = 0.004835, grid_max = 0.02, steps = 2)
        grid_param.setGrid(Param('gamma', par_min = 0.00000001), grid_avg = 0.09096, grid_min = 0.241531, grid_max = 0.30, steps = 2)
        grid_param.setGrid(Param('t1', par_min = 1), grid_avg = 4, grid_min = 3, grid_max = 6, steps = 2)
        grid_param.setGrid(Param('tgi2', par_min = 1), grid_avg = 23, grid_min = 7, grid_max = 15, steps = 2)
        grid_param.setGrid(Param('tgn2', par_min = 1), grid_avg = 18, grid_min = 14, grid_max = 25, steps = 2)
        grid_param.setGrid(Param('ta2', par_min = 1), grid_avg = 6, grid_min = 3, grid_max = 15, steps = 2)
        grid_param.setGrid(Param('Igs_t0', par_min = 0), grid_avg = 45, grid_min = 45, grid_max = 200, steps = 2)
        grid_param.setGrid(Param('Igci_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Igcn_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ias_t0', par_min = 0), grid_avg = 297, grid_min = 50, grid_max = 300, steps = 2)
        grid_param.setGrid(Param('M_t0', par_min = 0), grid_avg = 0, grid_min = 70, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ggci_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ggcn_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Gas_t0', par_min = 0), grid_avg = 0, grid_min = 100, grid_max = 500, steps = 1)
    elif grid_name == 'ita-broad':
        grid_param.setGrid(Param('rg', par_min = 0.0001), grid_avg = 0.22, grid_min = 0.08, grid_max = 0.65, steps = 3)
        grid_param.setGrid(Param('ra', par_min = 0.0001), grid_avg = 0.55, grid_min = 0.3, grid_max = 0.8, steps = 3)
        grid_param.setGrid(Param('alpha', par_min = 0.0001, par_max = 0.999), grid_avg = 0.68, grid_min = 0.18, grid_max = 0.95, steps = 3)
        grid_param.setGrid(Param('beta', par_min = 0.00000001, par_max = 0.999), grid_avg = 0.015, grid_min = 0.005, grid_max = 0.25, steps = 3)
        grid_param.setGrid(Param('beta_gcn', par_min = 0.00000001, par_max = 0.999), grid_avg = 0.015, grid_min = 0.005, grid_max = 0.25, steps = 3)
        grid_param.setGrid(Param('gamma', par_min = 0.00000001, par_max = 0.999), grid_avg = 0.067, grid_min = 0.015, grid_max = 0.95, steps = 3)
        grid_param.setGrid(Param('t1', par_min = 1), grid_avg = 4, grid_min = 4, grid_max = 6, steps = 2)
        grid_param.setGrid(Param('tgi2', par_min = 1), grid_avg = 12, grid_min = 15, grid_max = 25, steps = 2)
        grid_param.setGrid(Param('tgn2', par_min = 1), grid_avg = 20, grid_min = 15, grid_max = 25, steps = 2)
        grid_param.setGrid(Param('ta2', par_min = 1), grid_avg = 4, grid_min = 4, grid_max = 6, steps = 2)
        grid_param.setGrid(Param('Igs_t0', par_min = 0), grid_avg = 252, grid_min = 1, grid_max = 50, steps = 2)
        grid_param.setGrid(Param('Igci_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Igcn_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ias_t0', par_min = 0), grid_avg = 216, grid_min = 1, grid_max = 50, steps = 2)
        grid_param.setGrid(Param('M_t0', par_min = 0), grid_avg = 0, grid_min = 70, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ggci_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ggcn_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Gas_t0', par_min = 0), grid_avg = 0, grid_min = 100, grid_max = 500, steps = 1)
    elif grid_name == 'ita-broad_old':
        grid_param.setGrid(Param('rg', par_min = 0.0001), grid_avg = 0.22, grid_min = 0.08, grid_max = 0.65, steps = 3)
        grid_param.setGrid(Param('ra', par_min = 0.0001), grid_avg = 0.55, grid_min = 0.3, grid_max = 0.8, steps = 3)
        grid_param.setGrid(Param('alpha', par_min = 0.0001, par_max = 0.999), grid_avg = 0.68, grid_min = 0.18, grid_max = 0.95, steps = 3)
        grid_param.setGrid(Param('beta', par_min = 0.00000001, par_max = 0.999), grid_avg = 0.015, grid_min = 0.005, grid_max = 0.25, steps = 3)
        grid_param.setGrid(Param('beta_gcn', par_min = 0.00000001, par_max = 0.999), grid_avg = 0.015, grid_min = 0.005, grid_max = 0.25, steps = 3)
        grid_param.setGrid(Param('gamma', par_min = 0.00000001, par_max = 0.999), grid_avg = 0.067, grid_min = 0.015, grid_max = 0.95, steps = 3)
        grid_param.setGrid(Param('t1', par_min = 2, par_max = 40), grid_avg = 10, grid_min = 5, grid_max = 20, steps = 3)
        grid_param.setGrid(Param('tgi2', par_min = 3, par_max = 40), grid_avg = 20, grid_min = 12, grid_max = 35, steps = 3)
        grid_param.setGrid(Param('tgn2', par_min = 3, par_max = 40), grid_avg = 20, grid_min = 12, grid_max = 35, steps = 3)
        grid_param.setGrid(Param('ta2', par_min = 2, par_max = 20), grid_avg = 15, grid_min = 5, grid_max = 20, steps = 3)
        grid_param.setGrid(Param('Igs_t0', par_min = 0), grid_avg = 100, grid_min = 2, grid_max = 200, steps = 3)
        grid_param.setGrid(Param('Igci_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 100, steps = 1)
        grid_param.setGrid(Param('Igcn_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ias_t0', par_min = 0), grid_avg = 200, grid_min = 2, grid_max = 200, steps = 3)
        grid_param.setGrid(Param('M_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ggci_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ggcn_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Gas_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 300, steps = 1)
    elif grid_name == 'ita-broad_fixed':
        grid_param.setGrid(Param('rg', par_min = 0.000001), grid_avg = 0.10, grid_min = 0.03, grid_max = 0.40, steps = 3)
        grid_param.setGrid(Param('ra', par_min = 0.000001), grid_avg = 0.45, grid_min = 0.05, grid_max = 1.1, steps = 3)
        grid_param.setGrid(Param('alpha', par_min = 0.00000001), grid_avg = 0.78, grid_min = 0.15, grid_max = 0.95, steps = 3)
        grid_param.setGrid(Param('beta', par_min = 0.00000001), grid_avg = 0.04, grid_min = 0.003, grid_max = 0.2, steps = 3)
        grid_param.setGrid(Param('beta_gcn', par_min = 0.00000001), grid_avg = 0.04, grid_min = 0.003, grid_max = 0.2, steps = 3)
        grid_param.setGrid(Param('gamma', par_min = 0.00000001), grid_avg = 0.1, grid_min = 0.02, grid_max = 0.35, steps = 3)
        grid_param.setGrid(Param('t1', par_min = 1), grid_avg = 5, grid_min = 4, grid_max = 6, steps = 2)
        grid_param.setGrid(Param('tgi2', par_min = 1), grid_avg = 12, grid_min = 15, grid_max = 25, steps = 2)
        grid_param.setGrid(Param('tgn2', par_min = 1), grid_avg = 25, grid_min = 15, grid_max = 25, steps = 2)
        grid_param.setGrid(Param('ta2', par_min = 1), grid_avg = 4, grid_min = 4, grid_max = 6, steps = 1)
        grid_param.setGrid(Param('Igs_t0', par_min = 0), grid_avg = 100, grid_min = 1, grid_max = 10, steps = 2)
        grid_param.setGrid(Param('Igci_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Igcn_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ias_t0', par_min = 0), grid_avg = 42, grid_min = 1, grid_max = 10, steps = 2)
        grid_param.setGrid(Param('M_t0', par_min = 0), grid_avg = 0, grid_min = 70, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ggci_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ggcn_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Gas_t0', par_min = 0), grid_avg = 0, grid_min = 100, grid_max = 500, steps = 1)
    elif grid_name == 'ita-broad_fast':
        grid_param.setGrid(Param('rg', par_min = 0.0001), grid_avg = 0.22, grid_min = 0.08, grid_max = 0.8, steps = 2)
        grid_param.setGrid(Param('ra', par_min = 0.0001), grid_avg = 0.55, grid_min = 0.09, grid_max = 0.8, steps = 2)
        grid_param.setGrid(Param('alpha', par_min = 0.0001, par_max = 0.999), grid_avg = 0.68, grid_min = 0.18, grid_max = 0.95, steps = 2)
        grid_param.setGrid(Param('beta', par_min = 0.00000001, par_max = 0.999), grid_avg = 0.0072, grid_min = 0.0009, grid_max = 0.11, steps = 2)
        grid_param.setGrid(Param('beta_gcn', par_min = 0.00000001, par_max = 0.999), grid_avg = 0.0065, grid_min = 0.0009, grid_max = 0.11, steps = 1)
        grid_param.setGrid(Param('gamma', par_min = 0.00000001, par_max = 0.999), grid_avg = 0.067, grid_min = 0.015, grid_max = 0.95, steps = 2)
        grid_param.setGrid(Param('t1', par_min = 2, par_max = 40), grid_avg = 10, grid_min = 2, grid_max = 30, steps = 2)
        grid_param.setGrid(Param('tgi2', par_min = 3, par_max = 40), grid_avg = 15, grid_min = 3, grid_max = 35, steps = 2)
        grid_param.setGrid(Param('tgn2', par_min = 3, par_max = 40), grid_avg = 15, grid_min = 3, grid_max = 35, steps = 2)
        grid_param.setGrid(Param('ta2', par_min = 2, par_max = 20), grid_avg = 15, grid_min = 5, grid_max = 20, steps = 2)
        grid_param.setGrid(Param('Igs_t0', par_min = 0), grid_avg = 100, grid_min = 2, grid_max = 200, steps = 2)
        grid_param.setGrid(Param('Igci_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 100, steps = 1)
        grid_param.setGrid(Param('Igcn_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ias_t0', par_min = 0), grid_avg = 200, grid_min = 2, grid_max = 200, steps = 2)
        grid_param.setGrid(Param('M_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ggci_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ggcn_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Gas_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 300, steps = 1)
    elif grid_name == 'ita-broad_fast_fixed':
        grid_param.setGrid(Param('rg', par_min = 0.0001), grid_avg = 0.22, grid_min = 0.08, grid_max = 0.8, steps = 2)
        grid_param.setGrid(Param('ra', par_min = 0.0001), grid_avg = 0.55, grid_min = 0.09, grid_max = 0.8, steps = 2)
        grid_param.setGrid(Param('alpha', par_min = 0.0001, par_max = 0.999), grid_avg = 0.68, grid_min = 0.18, grid_max = 0.95, steps = 2)
        grid_param.setGrid(Param('beta', par_min = 0.00000001, par_max = 0.999), grid_avg = 0.0072, grid_min = 0.0009, grid_max = 0.11, steps = 2)
        grid_param.setGrid(Param('beta_gcn', par_min = 0.00000001, par_max = 0.999), grid_avg = 0.0065, grid_min = 0.0009, grid_max = 0.11, steps = 1)
        grid_param.setGrid(Param('gamma', par_min = 0.00000001, par_max = 0.999), grid_avg = 0.067, grid_min = 0.015, grid_max = 0.95, steps = 2)
        grid_param.setGrid(Param('t1', par_min = 2, par_max = 40), grid_avg = 10, grid_min = 2, grid_max = 30, steps = 2)
        grid_param.setGrid(Param('tgi2', par_min = 3, par_max = 40), grid_avg = 15, grid_min = 3, grid_max = 35, steps = 2)
        grid_param.setGrid(Param('tgn2', par_min = 3, par_max = 40), grid_avg = 15, grid_min = 3, grid_max = 35, steps = 2)
        grid_param.setGrid(Param('ta2', par_min = 2, par_max = 20), grid_avg = 15, grid_min = 5, grid_max = 20, steps = 2)
        grid_param.setGrid(Param('Igs_t0', par_min = 0), grid_avg = 100, grid_min = 2, grid_max = 200, steps = 2)
        grid_param.setGrid(Param('Igci_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 100, steps = 1)
        grid_param.setGrid(Param('Igcn_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ias_t0', par_min = 0), grid_avg = 200, grid_min = 2, grid_max = 200, steps = 2)
        grid_param.setGrid(Param('M_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ggci_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Ggcn_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 200, steps = 1)
        grid_param.setGrid(Param('Gas_t0', par_min = 0), grid_avg = 0, grid_min = 0, grid_max = 300, steps = 1)
    return grid_param