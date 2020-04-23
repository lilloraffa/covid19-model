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
    

    plt.plot(getattr(data_uff, var_name)[data_uff.i_start:], label = "Actual")
    plotOptModLine(var_name, opt_mod, 'tot', 'Model')
    #plotOptModLine(var_name, opt_mod, 'Igc_cum', 'Igc_cum')
    #plotOptModLine(var_name, opt_mod, 'Igc', 'Igc')
    #plotOptModLine(var_name, opt_mod, 'Gc_cum', 'Gc_cum')
    #plotOptModLine(var_name, opt_mod, 'M_cum', 'M_cum')
    #plt.ylim((0,20000))
    plt.xlabel('giorni')
    plt.ylabel('n. persone')
    plt.title('Actual vs Model: ' + var_dic[var_name])
    plt.legend()
    plt.show()

mod_var_dic = {
    'Igc_cum' : 'Total Infected',
    'Igc' : 'Currently Infected',
    'Igci_t': 'Curr. in Intensive Care',
    'Igcn_t': 'Curr. not in Intensive Care',
    'Gc_cum': 'Known Recovered',
    'M_cum': 'Known Dead'
}

def graphAllCompActBox(mod_name, opt_mod, data_uff, pred_days = 0):
    
    fig, axs = plt.subplots(3, 2, figsize=(12,15))
    i_start = opt_mod[mod_name]['i_start']
    period = opt_mod[mod_name]['period']
    
    
    var_name = 'dat_Igc_cum'
    var_label = mod_var_dic[var_name.replace('dat_', '')]
    
    axs[0, 0].plot(getattr(data_uff, var_name)[data_uff.i_start:], label = "Actual")
    axs[0, 0].plot((opt_mod[mod_name]['mod_data'][var_name])[i_start : i_start + period], label = "Model: " + var_name.replace('dat_', ''), color='C1')
    axs[0, 0].plot(range(period, period + pred_days), (opt_mod[mod_name]['mod_data'][var_name])[i_start + period: i_start + period + pred_days], label = "Model: " + var_name.replace('dat_', ''), color='C1', dashes=[6, 2])
    axs[0, 0].set_title('Actual vs Model: ' + var_label)

    var_name = 'dat_Igc'
    var_label = mod_var_dic[var_name.replace('dat_', '')]
    axs[0, 1].plot(getattr(data_uff, var_name)[data_uff.i_start:], label = "Actual")
    axs[0, 1].plot((opt_mod[mod_name]['mod_data'][var_name])[i_start : i_start + period], label = "Model: " + var_name.replace('dat_', ''), color='C1')
    axs[0, 1].plot(range(period, period + pred_days), (opt_mod[mod_name]['mod_data'][var_name])[i_start + period: i_start + period + pred_days], label = "Model: " + var_name.replace('dat_', ''), color='C1', dashes=[6, 2])
    axs[0, 1].set_title('Actual vs Model: ' + var_label)

    var_name = 'dat_Igci_t'
    var_label = mod_var_dic[var_name.replace('dat_', '')]
    axs[1, 0].plot(getattr(data_uff, var_name)[data_uff.i_start:], label = "Actual")
    axs[1, 0].plot((opt_mod[mod_name]['mod_data'][var_name])[i_start : i_start + period], label = "Model: " + var_name.replace('dat_', ''), color='C1')
    axs[1, 0].plot(range(period, period + pred_days), (opt_mod[mod_name]['mod_data'][var_name])[i_start + period: i_start + period + pred_days], label = "Model: " + var_name.replace('dat_', ''), color='C1', dashes=[6, 2])
    axs[1, 0].set_title('Actual vs Model: ' + var_label)

    var_name = 'dat_Igcn_t'
    var_label = mod_var_dic[var_name.replace('dat_', '')]
    axs[1, 1].plot(getattr(data_uff, var_name)[data_uff.i_start:], label = "Actual")
    axs[1, 1].plot((opt_mod[mod_name]['mod_data'][var_name])[i_start : i_start + period], label = "Model: " + var_name.replace('dat_', ''), color='C1')
    axs[1, 1].plot(range(period, period + pred_days), (opt_mod[mod_name]['mod_data'][var_name])[i_start + period: i_start + period + pred_days], label = "Model: " + var_name.replace('dat_', ''), color='C1', dashes=[6, 2])
    axs[1, 1].set_title('Actual vs Model: ' + var_label)

    var_name = 'dat_Gc_cum'
    var_label = mod_var_dic[var_name.replace('dat_', '')]
    axs[2, 0].plot(getattr(data_uff, var_name)[data_uff.i_start:], label = "Actual")
    axs[2, 0].plot((opt_mod[mod_name]['mod_data'][var_name])[i_start : i_start + period], label = "Model: " + var_name.replace('dat_', ''), color='C1')
    axs[2, 0].plot(range(period, period + pred_days), (opt_mod[mod_name]['mod_data'][var_name])[i_start + period: i_start + period + pred_days], label = "Model: " + var_name.replace('dat_', ''), color='C1', dashes=[6, 2])
    axs[2, 0].set_title('Actual vs Model: ' + var_label)

    var_name = 'dat_M_cum'
    var_label = mod_var_dic[var_name.replace('dat_', '')]
    axs[2, 1].plot(getattr(data_uff, var_name)[data_uff.i_start:], label = "Actual")
    axs[2, 1].plot((opt_mod[mod_name]['mod_data'][var_name])[i_start : i_start + period], label = "Model: " + var_name.replace('dat_', ''), color='C1')
    axs[2, 1].plot(range(period, period + pred_days), (opt_mod[mod_name]['mod_data'][var_name])[i_start + period: i_start + period + pred_days], label = "Model: " + var_name.replace('dat_', ''), color='C1', dashes=[6, 2])
    axs[2, 1].set_title('Actual vs Model: ' + var_label)

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

def graphR0(opt_mod, length = None):
    rg_period = opt_mod.rg_period
    if length is not None: rg_period = rg_period[0:length]
    ra_period = opt_mod.ra_period
    if length is not None: ra_period = ra_period[0:length]
    R0 = opt_mod.R0_t
    if length is not None: R0 = R0[0:length]

    fig = plt.figure(figsize = (10,7))

    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    fig.suptitle('')
    fig.suptitle('Virus infecting capability: R0, Rg, Ra')
    ax1.plot(R0)
    ax1.set_title('R0')
    ax2.plot(rg_period)
    ax2.set_title('Rg')
    ax3.plot(ra_period)
    ax3.set_title('Ra')
    plt.show()