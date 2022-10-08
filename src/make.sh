#!/bin/bash

resume_mode=0
num_gpus=1
round=8
num_experiments=4

# data
python make.py --mode data --run test --resume_mode 0 --num_gpus 4 --round 16 --num_experiments 1

# mvn-mean
python make.py --mode mvn-mean --run test --resume_mode 0 --num_gpus 4 --round 16 --num_experiments 1