import sys
import pickle
import pandas as pd
import numpy as np
import math
from datetime import datetime
import dateutil
import copy



# Function to get model data, in dictionary of np arrays
# mod: Model object to extract the data from
#dat_Igc_cum
def mod_dat(mod):

    dat_Igci_t = np.asarray(mod.Igci_t)
    dat_Igcn_t = np.asarray(mod.Igcn_t)
    dat_Ias_t = np.asarray(mod.Ias_t)
    dat_Igs_t = np.asarray(mod.Igs_t)
    dat_Igc = dat_Igci_t + dat_Igcn_t
    dat_I = dat_Igc + dat_Ias_t
    dat_NIs_t = np.asarray(mod.NIs_t)
    dat_G = np.asarray(mod.Gas_t) + np.asarray(mod.Ggci_t) + np.asarray(mod.Ggcn_t)
    dat_Gc = np.asarray(mod.Ggci_t) + np.asarray(mod.Ggcn_t)
    dat_M = np.asarray(mod.M_t)
    dat_G_cum = np.cumsum(dat_G)
    dat_Gc_cum = np.cumsum(dat_Gc)
    dat_M_cum = np.cumsum(dat_M)
    dat_Igc_cum = dat_Igc + dat_Gc_cum + dat_M_cum
    dat_Popi_t = np.asarray(mod.Popi_t)
    dat_R0_t = np.array(mod.R0_t)

    return {
        'dat_Igci_t' : dat_Igci_t,
        'dat_Igcn_t' : dat_Igcn_t,
        'dat_Ias_t' : dat_Ias_t,
        'dat_Igs_t' : dat_Igs_t,
        'dat_Igc' : dat_Igc,
        'dat_I' : dat_I,
        'dat_NIs' : dat_NIs_t,
        'dat_G' : dat_G,
        'dat_Gc' : dat_Gc,
        'dat_M' : dat_M,
        'dat_G_cum' : dat_G_cum,
        'dat_Gc_cum' : dat_Gc_cum,
        'dat_M_cum' : dat_M_cum,
        'dat_Igc_cum' : dat_Igc_cum,
        'dat_Popi_t': dat_Popi_t,
        'dat_R0_t': dat_R0_t,
    }

### Evolution Functions Functions

#utils
def getDataTS(t, data, def_val = 0): 
    #print(t, len(data), data)
    return data[t] if (t < len(data) and t > 0) else def_val

# f_dinamEvol: funzione che calcola il generico andamento di una variabile che ha evoluzione come: Vecchio + Nuovi - Uscenti
# Ias_t, Igci_t, Igcn_t
def f_dinamEvol(t, dat, Ndat, Udat):
    return max(0, dat[t] + Ndat[t+1] - Udat[t+1])




##### Evolutionary Temporal convention: X[t+1] = f(X[t])

## IFETTI SCONOSCIUTI

# f_NIs_t: funzione evolutiva del numero di nuovi infetti sconosciuti totali, dal tempo t al tempo t+1
# param:
#   t
#   rg: tasso contagio da infetto grave sconosciuto, al tempo t
#   ra: tasso contagio da infetto asintomatico sconosciuto, al tempo t
#   Igs_t: serie di Infettati gravi sconosciuti, al tempo t-1
#   Ias_t
def f_NIs_t(t, rg, ra, Igs_t, Ias_t):
    return max(0,rg * Igs_t[t] + ra * Ias_t[t])


## Ias_t

# f_NIas_t: funzione evolutiva del numero di nuovi infetti sconosciuti asintomatici, dal tempo t al tempo t+1
# param:
#   t
#   alpha
#   NIs_t
def f_NIas_t(t, alpha, NIs_t):
    NIs_tp1 = getDataTS(t+1, NIs_t)
    return max(0,(1 - alpha) * NIs_tp1)


# f_UIas_t: funzione evolutiva del numero di persone che smettono di essere sconosciuti asintomatici (U - Uscita), dal tempo t al tempo t+1
# param:
#   t
#   alpha
#   NIs_t
def f_UIas_t(t, ta2, NIas_t):
    NIs_tp1mta2 = getDataTS(t+1 - ta2, NIas_t)
    return max(0, NIs_tp1mta2)


# f_Ias_t: funzione evolutiva del numero di infetti asintomatici sconosciuti, dal tempo t al tempo t+1
# param:
#   t
#   ...
def f_Ias_t(t, Ias_t, NIas_t, UIas_t):
    return max(0, f_dinamEvol(t, Ias_t, NIas_t, UIas_t))



## Igs_t

# f_NIgs_t: funzione evolutiva del numero di nuovi infetti sconosciuti gravi, dal tempo t al tempo t+1
# param:
#   t
#   alpha
#   NIs_t
def f_NIgs_t(t, alpha, NIs_t):
    NIs_tp1 = getDataTS(t+1, NIs_t)
    return max(0, alpha * NIs_tp1)


# f_UIgs_t: funzione evolutiva del numero di persone che smettono di essere sconosciuti gravi (U - Uscita), dal tempo t al tempo t+1
# param:
#   t
#   alpha
#   NIs_t
def f_UIgs_t(t, t1, NIgs_t):
    NIgs_tp1mt1 = getDataTS(t+1 - t1, NIgs_t)
    return max(0, NIgs_tp1mt1)


# f_Igs_t: funzione evolutiva del numero di infetti gravi sconosciuti, dal tempo t al tempo t+1
# param:
#   t
#   ...
def f_Igs_t(t, Igs_t, NIgs_t, UIgs_t):
    return max(0, f_dinamEvol(t, Igs_t, NIgs_t, UIgs_t))




## Igci_t

# f_NIgci_t: funzione evolutiva del numero di nuovi infetti conosciuti gravi in terapia intensiva, dal tempo t al tempo t+1
# param:
#   t
#   alpha
#   NIs_t
def f_NIgci_t(t, gamma, UIgs_t):
    UIgs_tp1 = getDataTS(t+1, UIgs_t)
    return max(0, gamma * UIgs_tp1)


# f_Ggci_t: funzione evolutiva del numero di malati conosciuti gravi in terapia intensiva guariti, dal tempo t al tempo t+1
# param:
#   t
#   ...
def f_Ggci_t(t, t1, tgi2, beta, NIgci_t):
    NIgci_tp1mtgi2 = getDataTS(t+1 - tgi2, NIgci_t)
    return max(0, NIgci_tp1mtgi2 * ((1 - beta)**(tgi2 - t1)))


# f_Mgci_t: funzione evolutiva del numero di malati conosciuti gravi in terapia intensiva morti, dal tempo t al tempo t+1
# param:
#   t
#   ...
def f_Mgci_t(t, beta, Igci_t):
    return max(0, Igci_t[t] * beta )



# f_UIgci_t: funzione evolutiva del numero di persone che smettono di essere conosciuti gravi in terapia intensiva (U - Uscita), dal tempo t al tempo t+1
# param:
#   t
#   ...
def f_UIgci_t(t, Ggci_t, Mgci_t):
    return max(0, Ggci_t[t+1] + Mgci_t[t+1])


# f_Igci_t: funzione evolutiva del numero di infetti gravi conosciuti in terapia intensiva, dal tempo t al tempo t+1
# param:
#   t
#   ...
def f_Igci_t(t, Igci_t, NIgci_t, UIgci_t):
    return max(0, f_dinamEvol(t, Igci_t, NIgci_t, UIgci_t))




## Igcn_t

# f_NIgcn_t: funzione evolutiva del numero di nuovi infetti conosciuti gravi non in terapia intensiva, dal tempo t al tempo t+1
# param:
#   t
#   alpha
#   NIs_t
def f_NIgcn_t(t, gamma, UIgs_t):
    UIgs_tp1 = getDataTS(t+1, UIgs_t)
    return max(0, (1 - gamma) * UIgs_tp1)


# f_Ggcn_t: funzione evolutiva del numero di malati conosciuti gravi non in terapia intensiva guariti, dal tempo t al tempo t+1
# param:
#   t
#   ...
def f_Ggcn_t(t, t1, tgn2, beta_gcn, NIgcn_t):
    NIgcn_tp1mtgn2 = getDataTS(t+1 - tgn2, NIgcn_t)
    return max(0, NIgcn_tp1mtgn2 * ((1 - beta_gcn)**(tgn2 - t1)))


# f_Mgci_t: funzione evolutiva del numero di malati conosciuti gravi non in terapia intensiva morti, dal tempo t al tempo t+1
# param:
#   t
#   ...
def f_Mgcn_t(t, beta_gcn, Igcn_t):
    return max(0, Igcn_t[t] * beta_gcn )


# f_UIgci_t: funzione evolutiva del numero di persone che smettono di essere conosciuti gravi non in terapia intensiva (U - Uscita), dal tempo t al tempo t+1
# param:
#   t
#   ...
def f_UIgcn_t(t, Ggcn_t, Mgcn_t):
    return max(0, Ggcn_t[t+1] + Mgcn_t[t+1])


# f_Igci_t: funzione evolutiva del numero di infetti gravi conosciuti non in terapia intensiva, dal tempo t al tempo t+1
# param:
#   t
#   ...
def f_Igcn_t(t, Igcn_t, NIgcn_t, UIgcn_t):
    return max(0, f_dinamEvol(t, Igcn_t, NIgcn_t, UIgcn_t))


# f_Popi_t: funzione evolutiva della Popolazione Infettabile, dal tempo t-1 al tempo t
# Popi[t+1] = Pop_tot[0] - Igc_cum[t+1] - Igs[t+1] - Ias[t+1]
# param:
#   t
#   beta
def f_Popi_t(t, Pop_tot, Gas_t, Ggci_t, Ggcn_t, M_t, Igci_t, Igcn_t, Igs_t, Ias_t):
    return max(0, Pop_tot - sum(Gas_t) - sum(Ggci_t) - sum(Ggcn_t) - sum(M_t) - Igci_t[t] - Igcn_t[t] - Igs_t[t] - Ias_t[t])

def f_R0(t, rg_t, ra_t, t1, ta2, Igs_t, Ias_t):
    if (Igs_t[t] + Ias_t[t]) > 0:
        Igs_weight = Igs_t[t]/(Igs_t[t] + Ias_t[t])
        Ias_weight = Ias_t[t]/(Igs_t[t] + Ias_t[t])
    else:
        Igs_weight = 0
        Ias_weight = 0
    
    return Igs_weight * (((1+rg_t) ** t1) - 1) + Ias_weight * (((1+ra_t) ** ta2) - 1)

class Model:
    'Documentation'
    
    def __init__(self,
                 
                # rg: tasso di contagio infetti gravi, nell'unità di tempo
                rg = 0.75,

                # ra: tasso di contagio infetti asintomatici, nell'unità di tempo
                ra = 0.65,

                # alpha: probabilità che un nuovo infetto sia grave (sconosciuto), nell'unità di tempo
                alpha = 0.55,

                # beta: probabilità che un paziente grave infetto in terapia intensiva muoia, nell'unità di tempo
                beta = 0.30,
                 
                # beta_gcn: probabilità che un paziente grave infetto non in terapia intensiva muoia, nell'unità di tempo
                beta_gcn = 0.20,

                # gamma: probabilità che un paziente grave conosciuto vada in terapia intensiva, al tempo t1
                gamma = 0.15,

                # t1: tempo dall'infezione originaria (t0) in cui l'infettato diviene conosciuto grave
                t1 = 10,

                # tgi2: tempo dall'infezione originaria (t0) in cui l'infettato grave in terapia intensiva ritorna sano (guarisce)
                tgi2 = 20,

                # tgn2: tempo dall'infezione originaria (t0) in cui l'infettato grave non in terapia intensiva ritorna sano (guarisce)
                tgn2 = 15,

                # ta2: tempo dall'infezione originaria (t0) in cui l'infettato asintomatico ritorna sano (guarisce)
                ta2 = 15,
                
                # Igs_t: Infetti gravi non ancora conosciuti (sconosciuti), al tempo t
                Igs_t0 = 10,

                # Igci_t: Infetti gravi conosciuti ufficialmente in terapia intensiva, al tempo t
                Igci_t0 = 0,

                # Igcn_t: Infetti gravi conosciuti ufficialmente non in terapia intensiva, al tempo t
                Igcn_t0 = 0,

                # Ias_t: Infetti asintomatici non ancora conosciuti (sconosciuti), al tempo t
                Ias_t0 = 50,

                # M_t: Infetti gravi conosciuti in terapia intensiva morti, al tempo t
                M_t0 = 0,

                # Ggci_t: Infettati gravi conosciuti in terapia intensiva guariti, al tempo t
                Ggci_t0 = 0,

                # Ggcn_t: Infettati gravi conosciuti non in terapia intensiva guariti, al tempo t
                Ggcn_t0 = 0,

                # Gas_t: Infettati asintomatici non conosciuti (sconosciuti) guariti, al tempo t
                Gas_t0 = 0,
                 
                # Pop_tot: Totale Popolazione, al tempo t0
                Pop_tot = 60000000,
                 
                # rg_period: list of rg per unit of time
                rg_period = None,
                 
                # ra_period: list of ra per unit of time
                ra_period = None
                 
                ):
        
        ### Model Parameters
        # rg: tasso di contagio infetti gravi, nell'unità di tempo
        self.rg = rg

        # ra: tasso di contagio infetti asintomatici, nell'unità di tempo
        self.ra = ra

        # alpha: probabilità che un nuovo infetto sia grave (sconosciuto), nell'unità di tempo
        self.alpha = alpha

        # beta: probabilità che un paziente grave infetto in terapia intensiva muoia, nell'unità di tempo
        self.beta = beta
        
        # beta_gcn: probabilità che un paziente grave infetto non in terapia intensiva muoia, nell'unità di tempo
        self.beta_gcn = beta_gcn

        # gamma: probabilità che un paziente grave conosciuto vada in terapia intensiva, al tempo t1
        self.gamma = gamma

        # t: time of the current model
        self.t = 0

        # t1: tempo dall'infezione originaria (t0) in cui l'infettato diviene conosciuto grave
        self.t1 = t1

        # tgi2: tempo dall'infezione originaria (t0) in cui l'infettato grave in terapia intensiva ritorna sano (guarisce)
        self.tgi2 = tgi2

        # tgn2: tempo dall'infezione originaria (t0) in cui l'infettato grave non in terapia intensiva ritorna sano (guarisce)
        self.tgn2 = tgn2

        # ta2: tempo dall'infezione originaria (t0) in cui l'infettato asintomatico ritorna sano (guarisce)
        self.ta2 = ta2

        ### Model Variables
        # Igs_t: Infetti gravi non ancora conosciuti (sconosciuti), al tempo t
        self.Igs_t0 = Igs_t0
        self.Igs_t = [Igs_t0]

        # Igci_t: Infetti gravi conosciuti ufficialmente in terapia intensiva, al tempo t
        self.Igci_t0 = Igci_t0
        self.Igci_t = [Igci_t0]

        # Igcn_t: Infetti gravi conosciuti ufficialmente non in terapia intensiva, al tempo t
        self.Igcn_t0 = Igcn_t0
        self.Igcn_t = [Igcn_t0]

        # Ias_t: Infetti asintomatici non ancora conosciuti (sconosciuti), al tempo t
        self.Ias_t0 = Ias_t0
        self.Ias_t = [Ias_t0]

        # M_t: Infetti gravi conosciuti in terapia intensiva morti, al tempo t
        self.M_t0 = M_t0
        self.M_t = [M_t0]

        # Ggci_t: Infettati gravi conosciuti in terapia intensiva guariti, al tempo t
        self.Ggci_t0 = Ggci_t0
        self.Ggci_t = [Ggci_t0]

        # Ggcn_t: Infettati gravi conosciuti non in terapia intensiva guariti, al tempo t
        self.Ggcn_t0 = Ggcn_t0
        self.Ggcn_t = [Ggcn_t0]

        # Gas_t: Infettati asintomatici non conosciuti (sconosciuti) guariti, al tempo t
        self.Gas_t0 = Gas_t0
        self.Gas_t = [Gas_t0]
        
        # Pop_tot: Totale Popolazione, al tempo t0
        self.Pop_tot = Pop_tot
        
        # Popi_t: Totale Popolazione Infettabile, al tempo t
        self.Popi_t = [Pop_tot]
        
        # rg_period: array for each value of rg per unit of time
        self.rg_period = rg_period
        
        # ra_period: array for each value of ra per unit of time
        self.ra_period = ra_period
        
        ### Other model variables
        
        self.NIs_t = [0] # NIs_t: nuovi infettati sconosciuti
        self.NIas_t = [0] # NIs_t: nuovi infettati sconosciuti asintomatici
        self.UIas_t = [0] # NIs_t: nuovi infettati sconosciuti asintomatici
        self.NIgs_t = [0] # NIs_t: nuovi infettati sconosciuti asintomatici
        self.UIgs_t = [0] # NIs_t: nuovi infettati sconosciuti asintomatici
        self.NIgci_t = [0] # NIs_t: nuovi infettati sconosciuti asintomatici
        self.Mgci_t = [0] # NIs_t: nuovi infettati sconosciuti asintomatici
        self.UIgci_t = [0] # NIs_t: nuovi infettati sconosciuti asintomatici
        self.NIgcn_t = [0] # NIs_t: nuovi infettati sconosciuti asintomatici
        self.Mgcn_t = [0] # NIs_t: nuovi infettati sconosciuti asintomatici
        self.UIgcn_t = [0] # NIs_t: nuovi infettati sconosciuti asintomatici

        self.R0_t = [0] # NIs_t: nuovi infettati sconosciuti asintomatici
        
        
        self.params = {
            'rg' : rg,
            'ra' : ra,
            'alpha': alpha,
            'beta': beta,
            'beta_gcn': beta_gcn,
            'gamma': gamma,
            't1': t1,
            'tgi2': tgi2,
            'tgn2': tgn2,
            'ta2': ta2,
            'Igs_t0': Igs_t0,
            'Igci_t0': Igci_t0,
            'Igcn_t0': Igcn_t0,
            'Ias_t0': Ias_t0,
            'M_t0': M_t0,
            'Ggci_t0': Ggci_t0,
            'Ggcn_t0': Ggcn_t0,
            'Gas_t0': Gas_t0,
            'Pop_tot': Pop_tot,
            'rg_period': rg_period,
            'ra_period': ra_period
        }
        
        
    def stepforward(self):
        
        #0 rg_t e ra_t: tasso di infezione reale al tempo t
        rg = self.rg
        ra = self.ra
        if self.rg_period is not None:
            rg = self.rg_period[self.t]
        if self.ra_period is not None:
            ra = self.ra_period[self.t]
            
        rg_t = rg * (self.Popi_t[self.t] / self.Pop_tot)
        ra_t = ra * (self.Popi_t[self.t] / self.Pop_tot)
        
        #1 Calcolo nuovi infettati sconosciuti totali (gravi e asintomatici)
        self.NIs_t.append(f_NIs_t(self.t, rg_t, ra_t, self.Igs_t, self.Ias_t))
        
        #2 Calcolo variabili asintomatici per sconosciuti 
        self.NIas_t.append(f_NIas_t(self.t, self.alpha, self.NIs_t))
        self.UIas_t.append(f_UIas_t(self.t, self.ta2, self.NIas_t))
        self.Gas_t.append(self.UIas_t[self.t+1])
        self.Ias_t.append(f_Ias_t(self.t, self.Ias_t, self.NIas_t, self.UIas_t))
        
        #3 Calcolo variabili per gravi sconosciuti
        self.NIgs_t.append(f_NIgs_t(self.t, self.alpha, self.NIs_t))
        self.UIgs_t.append(f_UIgs_t(self.t, self.t1, self.NIgs_t))
        self.Igs_t.append(f_Igs_t(self.t, self.Igs_t, self.NIgs_t, self.UIgs_t)) 
        
        #4 Calcolo variabili per gravi conosciuti in terapia intensiva
        self.NIgci_t.append(f_NIgci_t(self.t, self.gamma, self.UIgs_t))
        self.Ggci_t.append(f_Ggci_t(self.t, self.t1, self.tgi2, self.beta, self.NIgci_t))
        self.Mgci_t.append(f_Mgci_t(self.t, self.beta, self.Igci_t))
        self.UIgci_t.append(f_UIgci_t(self.t, self.Ggci_t, self.Mgci_t))
        self.Igci_t.append(f_Igci_t(self.t, self.Igci_t, self.NIgci_t, self.UIgci_t)) 
        
        #5 Calcolo variabili per gravi conosciuti in terapia intensiva
        self.NIgcn_t.append(f_NIgcn_t(self.t, self.gamma, self.UIgs_t))
        self.Ggcn_t.append(f_Ggcn_t(self.t, self.t1, self.tgn2, self.beta_gcn, self.NIgcn_t))
        self.Mgcn_t.append(f_Mgcn_t(self.t, self.beta_gcn, self.Igcn_t))
        self.UIgcn_t.append(f_UIgcn_t(self.t, self.Ggcn_t, self.Mgcn_t))
        self.Igcn_t.append(f_Igcn_t(self.t, self.Igcn_t, self.NIgcn_t, self.UIgcn_t)) 
            
        #6 Popi_t: Totale Popolazione infettabile
        self.Popi_t.append(f_Popi_t(self.t, self.Pop_tot, self.Gas_t, self.Ggci_t, self.Ggcn_t, self.M_t, self.Igci_t, self.Igcn_t, self.Igs_t, self.Ias_t))
        
        #7 Calc totale morti conosciuti
        self.M_t.append(self.Mgci_t[self.t+1] + self.Mgcn_t[self.t+1])

        #8 Calc R0
        self.R0_t.append(f_R0(self.t, rg_t, ra_t, self.t1, self.ta2, self.Igs_t, self.Ias_t))
        
        self.t = self.t + 1
        
        
    def modstats(self, t, adv_viz = False):
        print('stats - t = %d \t I: %.2f \t Igc: %.2f \t Igcn: %.2f \t Igci: %.2f \t G: %.2f \t M: %.2f' %(
            t,
            (self.Ias_t[t] + self.Igs_t[t] + self.Igci_t[t] + self.Igcn_t[t]), 
            (self.Igci_t[t] + self.Igcn_t[t]),
            self.Igcn_t[t],
            self.Igci_t[t],
            (self.Ggci_t[t] + self.Ggcn_t[t] + self.Gas_t[t]),
            self.M_t[t]
         )
        )
        
    
    # Model run for n periods
    def run(self, n):
        #self.modstats(self.t)
        if self.rg_period is not None:
            rg_len = len(self.rg_period)
            rg_initial = [0 for x in range(0, rg_len + n)]
            self.rg_period = list(map(lambda x, i: self.rg_period[i] if i in range(0, rg_len) else self.rg_period[rg_len - 1], rg_initial, range(0,len(rg_initial))))
        if self.ra_period is not None:
            ra_len = len(self.ra_period)
            ra_initial = [0 for x in range(0, ra_len + n)]
            self.ra_period = list(map(lambda x, i: self.ra_period[i] if i in range(0, ra_len) else self.ra_period[ra_len - 1], ra_initial, range(0,len(ra_initial))))
        
        for i in range (0,n):
            self.stepforward()