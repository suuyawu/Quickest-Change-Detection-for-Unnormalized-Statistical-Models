#!/bin/bash

resume_mode=0
num_gpus=1
round=8
num_experiments=4

# data
python make.py --mode data --run test --resume_mode 0 --num_gpus 4 --round 16 --num_experiments 1

# mvn-mean
python make.py --mode mvn-mean --run test --resume_mode 0 --num_gpus 4 --round 16 --num_experiments 1

# mvn-mean-arl
python make.py --mode mvn-mean-arl --run test --resume_mode 0 --num_gpus 4 --round 16 --num_experiments 1

# mvn-mean-lambda
python make.py --mode mvn-mean-lambda --run test --resume_mode 0 --num_gpus 4 --round 16 --num_experiments 1

# mvn-mean-noise
python make.py --mode mvn-mean-noise --run test --resume_mode 0 --num_gpus 4 --round 16 --num_experiments 1

# mvn-logvar
python make.py --mode mvn-logvar --run test --resume_mode 0 --num_gpus 4 --round 16 --num_experiments 1

# mvn-logvar-arl
python make.py --mode mvn-logvar-arl --run test --resume_mode 0 --num_gpus 4 --round 16 --num_experiments 1

# mvn-logvar-lambda
python make.py --mode mvn-logvar-lambda --run test --resume_mode 0 --num_gpus 4 --round 16 --num_experiments 1

# mvn-logvar-noise
python make.py --mode mvn-logvar-noise --run test --resume_mode 0 --num_gpus 4 --round 16 --num_experiments 1

# exp-tau
python make.py --mode exp-tau --run test --resume_mode 0 --num_gpus 4 --round 16 --num_experiments 1

# exp-tau-arl
python make.py --mode exp-tau-arl --run test --resume_mode 0 --num_gpus 4 --round 16 --num_experiments 1

# exp-tau-lambda
python make.py --mode exp-tau-lambda --run test --resume_mode 0 --num_gpus 4 --round 16 --num_experiments 1

# exp-tau-noise
python make.py --mode exp-tau-noise --run test --resume_mode 0 --num_gpus 4 --round 16 --num_experiments 1

# rbm-W
python make.py --mode rbm-W --run test --resume_mode 0 --num_gpus 4 --round 16 --num_experiments 1

# rbm-W-arl
python make.py --mode rbm-W-arl --run test --resume_mode 0 --num_gpus 4 --round 16 --num_experiments 1

# rbm-W-lambda
python make.py --mode rbm-W-lambda --run test --resume_mode 0 --num_gpus 4 --round 16 --num_experiments 1

# rbm-W-noise
python make.py --mode rbm-W-noise --run test --resume_mode 0 --num_gpus 4 --round 16 --num_experiments 1