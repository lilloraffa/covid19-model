## Model

The model describes the evolution in time of a society in which an initial number of people affected by the virus is 
introduced. At each point in time, every person in the society is in one of the following categories:

1. $Ias_t$: unknown asymptomatic, people that will not suffer from the severe consequences of the virus and will 
continue their normal life. They will recover from the virus after $t2_{as}$ periods, becoming $Gas_t$.
2. $Igs_t$: unknown seriously affected, people that are currently not showing symptoms but will be hit by the serious
consequences of the virus after $t1$ periods, becoming either $Igci_t$ with probability $\gamma$ or $Igcn_t$ with 
probability $1 - \gamma$.
3. $Igci_t$: known seriuos affected in intensive care, they may die during each period of the hospitalization with 
probability $\beta_{gci}$, or, if they don't die, they will recover after $t2_{gi}$ periods becoming $Gci_t$.
4. $Igcn_t$: known serious affected but not in intensive care, they also may die during each period with probaility
$\beta_{gcn}$, or, alternatively, they will recover if they survive after $t2_{gn}$ periods becoming $Gcn_t$.
5. $Popi_t$: people that have not been affected by the virus and so can be affected by an infected person. Once 
infected, the person can become an unknown serious affected ($Igs_t$) with probability $\alpha$, or an unknown 
asymptomatic ($Ias_t$) with probability $1 - \alpha$

### Evolutionary equations
The equations that describe the evolution of each variable in time are desribed below. We wrote the equation with the 
following convention, to make it more readible: the prefix $N$ is used to descrive people that are new to the category 
$N$ refers to, while $U$ is used in the same fashion for those that exited the category in which they where before.

$$ Ias_{t+1} = Ias_{t} + NIas_{t+1} - UIas_{t+1}$$

where $NIas_{t+1}$ is the number of people been infected by either $Ias_t$ or $Igs_t$ during period $t$ and become new
asymptomatic infected at time $t+1$, and $UIas_{t+1}$ are those that recovered during/after time $t$

$$ NIas_{t+1} = (1 - \alpha) (rg_t Igs_t + ra_t Ias_t)$$

$$ UIas_{t+1} = Gas_{t+1} = NIas_{t+1 - t2_{as}}$$


$$ Igs_{t+1} = Igs_{t} + NIgs_{t+1} - UIgs_{t+1}$$

$$ NIgs_{t+1} = \alpha (rg_t Igs_t + ra_t Ias_t)$$

$$ UIgs_{t+1} = NIgs_{t+1 - t1}$$


$$ Igci_{t+1} = Igci_t + NIgci_{t+1} - UIgci_{t+1}$$

$$ NIgci_{t+1} = \gamma UIgs_{t+1}$$

$$ Uigci_{t+1} = Ggci_{t+1} + Mgci_{t+1}$$

$$ Ggci_{t+1} = NIgci_{t+1-t2_{gi}} (1-\beta)^{(t2_{gi} - t1)}$$

$$ Mgci_{t+1} = \beta Igci_t$$


$$ Igcn_{t+1} = Igcn_t + NIgcn_{t+1} - UIgcn_{t+1}$$

$$ NIgcn_{t+1} = (1 - \gamma) UIgs_{t+1}$$

$$ UIgcn_{t+1} = GIgcn_{t+1} + Mgcn_{t+1}$$

$$ Ggcn_{t+1} = NIgcn_{t+1 - t2_{gn}} (1 - \beta_{gcn})^{(t2_{gn} - t1)}$$

$$ Mgcn_{t+1} = \beta_{gcn} Igcn_{t}$$

$$ Igc_t = Igci_t + Igcn_t$$

Finally, $rg_t$ and $ra_t$ can be seen as a way to calculate what in medical theory is called $r_0$. They represent the 
number of people that will be infected in a period of time by an unknown seriously affected person and by an unknown 
asymptomatic, respectively. $rg_t$ and $ra_t$ are not constant parameters, as they vary due to environmental conditions: 
the higher are the chances of contacts between individuals, for instance, the higher they will be. From a mathematical
perspective, they represents an important closure condition: we'll make them depened on the population not yet affected,
 and this will make the model not exploding as time goes.

 $$ rg_t = rg \frac{Popi_t}{Pop}$$
 $$ ra_t = ra \frac{Popi_t}{Pop}$$

 where $Pop$ is the total population at time 0. In other words, you can see $rg$ and $ra$ as $rg_{t0}$ $ra_{t0}$, where 
 $t0$ is the starting time when the first infected person is introduced in the society.

 ### Model characteristics and parameter sensitivity
 The model is highly dependent on exogenous parameters: here we describe how the most important variable evolve in time
 given changes in the exogenous parameters. 

 ![Param Sensitivity: rg](./img/model_param_sensitivity_rg.png)
 ![Param Sensitivity: rg](./img/model_param_sensitivity_rg.png)



## Calibration
In order to calibrate the model, we proceed with an extensive grid search on all parameters and initial conditions. The 
procedure we developed consists on three different grid searches, that will be described below, each of which follows 
this path:

1. create a grid of parameters
2. run the model for every combination of parameters in the grid
3. compare the results of the model to actual data coming from the Dipartimento della Protezione Civile Italiana, and 
calculate an error measure (more on this later)
4. select an optimal model for each error calculated: we calculate errors on the number of total infected people since the beginning of the infection $err^{Igc_{cum}}$, number of people currently infected $err^{Igc}$, total number of deaths
$err^{M_{cum}}$, total number of known recovered peolpe $err^{Gc_{cum}}$ and the average of all the above $err^{tot}$

The calibration steps mentioned above are the following:

1. Initial grid search: create a grid of all parameters and find optimal models
2. Parameter fine tuning: starting from the optimal parameters from step 1, create a grid with +- a percentage increase of the optimal parameters and find optimal models
3. Window Search: we divide the periods in windows of fixed length (7 periods), and for each of them we allow the $rg$ and $ra$ parameter to change (like in step 2), so to allow for changing conditions in the infection rate. This will try to find changes in exogenous conditions affecting the infection rate, like social restriction measures taken by the 
government.


## Results
Here are the results of a calibration performed on Italian data at national level, updated as of March 28th. Differences
between the model and the window optimized one will be also presented.

DISCLAIMER: this is a quite preliminary version of the model, so its results are still under investigation. In the next days/weeks I'll be working on it and more robust analysis will be shared. For this reasons, please consider the 
following analysis only a pure academic exercise. For the same reasons, I'm not investing time in describing the results, 
but I'm only showing them as they are, also to get feedbacks that are always very very wellcome.

![Model curves](./img/Mod_v9_opt_tot_modcurves_20200328.png)
![Opt Window Model curves](./img/Mod_v9_optwindow_tot_modcurves_20200328.png)

| Parameter | Model     | Opt Window|
|-----------|----------:|----------:|
| rg        | 0.155     | 0.153     |
| ra        | 0.888     | 0.870     |
| alpha     | 0.599     | 0.599     |
| beta      | 0.011     | 0.011     |
| beta_gcn  | 0.009     | 0.009     |
| gamma     | 0.064     | 0.064     |
| t1        | 1         | 1         |
| tgi2      | 15        | 15        |
| tgn2      | 15        | 15        |
| ta2       | 103       | 103       |
| Igs_t0    | 156       | 156       |


![Model Stats](./img/Mod_v9_opt_tot_modcurves_20200328.png)
![Opt Window Model Stats](./img/Mod_v9_optwindow_tot_modcurves_20200328.png)

| Peak Var                                  | Model             | Opt Window        |
|-------------------------------------------|------------------:|------------------:|
| Total Infected (Igc_cum)                  | 9.85M @ Sep 08    | 8.72M @ Sep 08    |
| Daily Infected (Igc)                      | 2.76M @ May 28    | 2.14M @ Jun 03    |
| Daily Infected Int. Care (Igci)           |  175K @ May 28    |  136K @ Jun 02    |
| Daily Deaths (M)                          |   26M @ May 29    |   20K @ Jun 04    |
| Total Deaths (M_cum)                      | 1.27M @ Aug 04    | 1.12M @ Aug 20    |



Finally, here is a comparison between the model, optimal window model and actual data for key variables.

![Comparison: Igc_cum](./img/Mod_v9_both_tot_comp_Igc-cum_20200328.png)
![Comparison: Igc](./img/Mod_v9_both_tot_comp_Igc_20200328.png)
![Comparison: Ggc_cum](./img/Mod_v9_both_tot_comp_Ggc-cum_20200328.png)
![Comparison: M_cum](./img/Mod_v9_both_tot_comp_M-cum_20200328.png)