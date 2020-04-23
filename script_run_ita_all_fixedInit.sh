#!/bin/bash

clear
source $(conda info --base)/etc/profile.d/conda.sh
conda activate data
python python/calibration.py --model_name=Model-v5_err1abs-fixedInit --param_fixed --run_partial 0 10 \
    --param_exclude_finetuner Igs_t0, Ias_t0\
    --param_exclude_window rg ra Igs_t0, Ias_t0 \
    --model_stepsforward=200 &

python python/calibration.py --model_name=Model-v5_err1abs-fixedInit --param_fixed --run_partial 10 -999 \
    --param_exclude_finetuner Igs_t0, Ias_t0\
    --param_exclude_window rg ra Igs_t0, Ias_t0 \
    --model_stepsforward=200 &