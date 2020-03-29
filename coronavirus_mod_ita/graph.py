import matplotlib.pyplot as plt



var_dic = {
    'dat_Igc_cum': "Infetti Gravi Tot Cumulati",
    'dat_Igc': "Infetti Attualmente Gravi",
    'dat_Igci_t': "Infetti Attualmente Gravi in Terapia Intensiva",
    'dat_Igcn_t': "Infetti Attualmente Gravi non in Terapia Intensiva",
    'dat_Gc_cum': "Infetti Gravi Guariti Cumulati",
    'dat_M_cum': "Morti Cumulati"
}
optmod_dic = {
    'tot': 'Errrore totale',
    'Igc_cum': 'Errore su Infetti totali cumulati',
    'Igc': 'Errore su Infetti attuali',
    'Gc_cum': 'Errore su Guariti cumulati',
    'M_cum': 'Errore su Morti cumulati'
}

pred_days = 300

def plotOptModLine(var_name, opt_mod, optmod_name, label_val='', pred_days=0):
    plt.plot(range(0,opt_mod[optmod_name]['period']+pred_days), (opt_mod[optmod_name]['mod_data'][var_name])[opt_mod[optmod_name]['i_start']:opt_mod[optmod_name]['i_start'] + opt_mod[optmod_name]['period']+pred_days], label = label_val)

def graphAllCompAct(var_name, opt_mod, data_uff):
    #var_name = 'dat_Igc_cum'
    #opt_mod = mod_optimizer.opt_model


    plt.figure(figsize = (10,7))
    

    plt.plot(getattr(data_uff, var_name)[data_uff.i_start:], label = "UFF")
    plotOptModLine(var_name, opt_mod, 'tot', 'tot')
    plotOptModLine(var_name, opt_mod, 'Igc_cum', 'Igc_cum')
    plotOptModLine(var_name, opt_mod, 'Igc', 'Igc')
    plotOptModLine(var_name, opt_mod, 'Gc_cum', 'Gc_cum')
    plotOptModLine(var_name, opt_mod, 'M_cum', 'M_cum')
    #plt.ylim((0,20000))
    plt.xlabel('giorni')
    plt.ylabel('n. persone')
    plt.title('Actual vs Model: ' + var_dic[var_name])
    plt.legend()
    plt.show()
    
def graphOptAll(optmod_name, opt_mod, data_uff, var_name_uff = 'dat_Igc_cum', pred_days=0):

    plt.figure(figsize = (10,7))
    plt.plot(getattr(data_uff, var_name_uff)[data_uff.i_start:], label = "UFF: " + var_name_uff)
    plotOptModLine('dat_Igc_cum', opt_mod, optmod_name, 'dat_Igc_cum', pred_days)
    plotOptModLine('dat_Igc', opt_mod, optmod_name, 'dat_Igc', pred_days)
    plotOptModLine('dat_Igci_t', opt_mod, optmod_name, 'dat_Igci_t', pred_days)
    plotOptModLine('dat_Gc_cum', opt_mod, optmod_name, 'dat_Gc_cum', pred_days)
    plotOptModLine('dat_Gc', opt_mod, optmod_name, 'dat_Gc', pred_days)
    plotOptModLine('dat_G', opt_mod, optmod_name, 'dat_G', pred_days)
    plotOptModLine('dat_M_cum', opt_mod, optmod_name, 'dat_M_cum', pred_days)
    plotOptModLine('dat_M', opt_mod, optmod_name, 'dat_M', pred_days)
    
    if pred_days>0:
        period = opt_mod[optmod_name]['period']
        max_line = (opt_mod[optmod_name]['mod_data']['dat_Igc_cum'])[opt_mod[optmod_name]['i_start']:opt_mod[optmod_name]['i_start'] + opt_mod[optmod_name]['period']+pred_days].max()
        plt.plot([period-1, period-1], [0, max_line], 'k-', lw=2)
    #plt.ylim((0,20000))
    plt.xlabel('giorni')
    plt.ylabel('n. persone')
    plt.title('Model: %s - Predictions' %(optmod_name))
    plt.legend()
    plt.show()
    
def predNextDays(optmod_name, opt_mod, var_name, pred_days):
    pred = (opt_mod[optmod_name]['mod_data'][var_name])[opt_mod[optmod_name]['i_start'] + opt_mod[optmod_name]['period'] -1 :opt_mod[optmod_name]['i_start'] + opt_mod[optmod_name]['period']+pred_days]
    print("Mod: %s \t Next days: %s: \t %s" %(optmod_name, var_name, str([int(x) for x in pred])))
    print("Mod: %s \t Variazion: %s: \t %s" %(optmod_name, var_name, str([int(x) for x in (pred[1:len(pred)] - pred[0:len(pred)-1])])))
    

def graphWindowComp(var_name, data_uff, opt_mod, opt_mod_window, model_name):
    plt.figure(figsize = (10,7))
    plt.plot(getattr(data_uff, var_name)[data_uff.i_start:], label = "UFF")
    plotOptModLine(var_name, opt_mod, model_name, 'Opt')
    plotOptModLine(var_name, opt_mod_window, model_name, 'Opt_window')
    plt.xlabel('giorni')
    plt.ylabel('n. persone')
    plt.title('Actual vs ModelOpt vs OptWindows: ' + var_dic[var_name])
    plt.legend()
    plt.show()