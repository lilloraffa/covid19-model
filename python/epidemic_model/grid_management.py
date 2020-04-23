import math
import numpy as np
from .model import * 

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
            if (steps_min>0 and steps_max>0 and (val_avg - val_min)>0 and (val_max - val_avg)>0):
                gridList = np.arange(val_min, val_avg, (val_avg - val_min)/(steps_min)).tolist()
                gridList = gridList + np.arange(val_avg, val_max, (val_max - val_avg)/(steps_max)).tolist()
                gridList.append(val_max)
            else:
                gridList = [val_min, val_avg, val_max]
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

def getParamList(param_list_init = None, exclude = None):
        mod_generic = Model()
        exclude_param = ['Pop_tot', 'Igci_t0', 'Igcn_t0', 'M_t0', 'Ggci_t0', 'Ggcn_t0', 'Gas_t0']
        param_list = mod_generic.params.keys()
        if param_list_init is not None and param_list_init != []:
            param_list = param_list_init
            
        if exclude is not None:
            exclude_param = exclude_param + exclude

        return [x for x in param_list if x not in exclude_param], exclude_param