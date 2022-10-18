import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import models
import numpy as np
from config import cfg, process_args
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, resume, collate
from logger import make_logger




if __name__ == "__main__":
    num_trials = 100
    num_pre = 9500
    data = torch.randn(95000, 2)
    randidx = torch.randint(0, data.shape[0], (num_trials, num_pre))
    print(randidx.shape)
    x = data[randidx]
    print(x.shape)
