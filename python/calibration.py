from init_model import * 
from epidemic_model import *
from data_mgmt import *
from datetime import datetime
import argparse


parser = argparse.ArgumentParser(description='Covid-19 model calibration')
parser.add_argument("--macro_geo", default='ita', 
    help="""Code of the macro area containing the single geo used for estimation. It used to manage initial dataset 
    and grid set up.""")
parser.add_argument("--model_name", default='Model-v1_err1abs_debug', 
    help="""Model name. This will also be the name of the object filed saved after the calibration procedure.""")
parser.add_argument("--model_stepsforward", type=int, default=250, help="Number of steps that will be run for each model.")
parser.add_argument("--run_partial", nargs='+', type=int, default=None, help="Base path for the entire project.")
parser.add_argument("--run_geolist", nargs='+', default=None, help="List of geocode to run the model apon.")
parser.add_argument("--run_fast", action="store_true", default=False, help="Base path for the entire project.")
parser.add_argument("--param_exclude_grid", nargs='+', default=None, help="List of param name to exclude from the initial grid search.")
parser.add_argument("--param_exclude_finetuner", nargs='+', default=None, help="List of param name to exclude from parameter FineTuner algorithm.")
parser.add_argument("--param_exclude_window", nargs='+', default=None, help="List of param name to exclude from window optimization procedure.")
parser.add_argument("--param_fixed", action="store_true", default=False, help="If you want to run a model with fixed params in grid search.")
parser.add_argument("--run_infloop", action="store_true", default=False, help="Use this paraeter if you want the loop to never end")
parser.add_argument("--path_stored_models", default="saved_models/", help="Path where previously calibrated models are stored.")
parser.add_argument("--path_base", default="", help="Base path for the entire project.")

args = parser.parse_args()

#Initialize list of geographies to analyze



def start():
    __RELATIVE_PATH = args.path_base
    __SAVED_MOD_PATH = args.path_stored_models
    __MACRO_GEO = args.macro_geo
    __MOD_NAME = args.model_name
    __STEPFORWARD = args.model_stepsforward
    __INIT_GRID_TYPE = '_fast' if args.run_fast else ''
    __INIT_GRID_TYPE = __INIT_GRID_TYPE+'_fixed' if args.param_fixed else __INIT_GRID_TYPE
    __PARAM_EXCLUDE_GRID = args.param_exclude_grid if args.param_exclude_grid is not None else []
    __PARAM_EXCLUDE_FINETUNER = args.param_exclude_finetuner if args.param_exclude_finetuner is not None else []
    __PARAM_EXCLUDE_WINDOW = args.param_exclude_window if args.param_exclude_window is not None else []

    print("*********** Starting Calibration ***************")
    print(__MACRO_GEO, __MOD_NAME, __INIT_GRID_TYPE)

    #Initialize list of geographies to analyze
    refreshData(rel_path = __RELATIVE_PATH)
    import_data, opt_mod_obj = getData(geo = __MACRO_GEO, obj_path = __SAVED_MOD_PATH, mod_name = __MOD_NAME)
    geo_list = import_data.data['geo_code'].unique()
    geo_list.sort() 
    geo_list = geo_list.tolist() + ['aggr']

    index_init = 0
    index_end = len(geo_list)
    run_partial = args.run_partial
    if run_partial is not None:
        if type(run_partial) is not list:
            run_partial = [run_partial]
        if len(run_partial)==2:
            index_init = run_partial[0]
            if run_partial[1] != -999: index_end = run_partial[1]
        elif len(run_partial)==1 and run_partial[0] is not None:
            index_end = run_partial[0]

    if args.run_geolist is not None:
        geo_list = args.run_geolist
    #geo_list = ['aggr']


    for geo in geo_list[index_init:index_end]:
        grid_param = None
        data = import_data.data
        data_uff = None
        if geo != "aggr":
            data_uff = ActualData(data[data['geo_code'] == geo].reset_index())
            init_grid_name = '-broad'
            pop = import_data.data_anagr[import_data.data_anagr['geo_code'] == geo]['pop'].unique()
            name = import_data.data_anagr[import_data.data_anagr['geo_code'] == geo]['geo_name'].unique()
            print("**** Running model for region: %d, %s, %d" %(geo, name, pop))
        else:
            data_uff = ActualData(import_data.data_aggr)
            init_grid_name = '-aggr'
            pop = import_data.data_anagr['pop'].sum()
            name = 'National'
            print("**** Running model for National: %d" %(pop))

        opt_mod_prev = None
        opt_mod_prev_window = None
        if geo in opt_mod_obj['model'].keys():
            #print("Already Exists")
            
            opt_mod_prev = opt_mod_obj['model'][geo]['opt']
            opt_mod_prev_window = opt_mod_obj['model'][geo]['opt_window']
            #print(opt_mod_prev['mod'].params)
            if opt_mod_prev_window['tot']['mod'].params['rg_period'] is not None:
                __PARAM_EXCLUDE_FINETUNER = __PARAM_EXCLUDE_FINETUNER + ['rg']
                __PARAM_EXCLUDE_GRID = __PARAM_EXCLUDE_GRID + ['rg']
            if opt_mod_prev_window['tot']['mod'].params['ra_period'] is not None:
                __PARAM_EXCLUDE_FINETUNER = __PARAM_EXCLUDE_FINETUNER + ['ra']
                __PARAM_EXCLUDE_GRID = __PARAM_EXCLUDE_GRID + ['ra']

            grid_param = get_grid_param(opt_mod_prev_window['tot']['mod'].params, delta_mult = 1.3, exclude=__PARAM_EXCLUDE_GRID)
        else:
            #print("New Exists: " + __MACRO_GEO + init_grid_name + __INIT_GRID_TYPE)
            grid_param = load_grid(__MACRO_GEO + init_grid_name + __INIT_GRID_TYPE)
        #print("__PARAM_EXCLUDE_WINDOW: " + str(__PARAM_EXCLUDE_WINDOW))
        print(grid_param.paramGrid)
        mod_optimizer = ModelOptimizer(data_uff, grid_param, __STEPFORWARD, sync_data_perc = 0.3, Pop_tot=pop,
            opt_model = opt_mod_prev,
            opt_model_window = opt_mod_prev_window,
            exclude_param_finetuner = __PARAM_EXCLUDE_FINETUNER,
            exclude_param_window = __PARAM_EXCLUDE_WINDOW)
        mod_optimizer.start()
        
        opt_mod_obj['model'][geo] = {
            'name': name, 
            'opt': mod_optimizer.opt_model, 
            'opt_window': mod_optimizer.opt_model_window, 
            'date_update': datetime.now(),
            'date_last_actual': data['date'].max()
        }
        saveOptMod(opt_mod_obj, data_uff.date.max(), path = __RELATIVE_PATH + __SAVED_MOD_PATH, mod_name = __MOD_NAME, 
            bDataUpdateTime = False,
            bRunDatetime = False)

    saveOptMod(opt_mod_obj, data_uff.date.max(), path = __RELATIVE_PATH + __SAVED_MOD_PATH, mod_name = __MOD_NAME,
        bRunDatetime = False)


while args.run_infloop:
    print("*******inf_loop")
    start()
else:
    start()