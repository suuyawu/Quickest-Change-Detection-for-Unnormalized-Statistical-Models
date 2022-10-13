import argparse
import os
import time
import torch
import torch.backends.cudnn as cudnn
import models
from collections import defaultdict
from config import cfg, process_args
from data import fetch_dataset, make_data_loader
from utils import save, to_device, process_control, process_dataset, resume, collate, make_params, make_footprint
from modules import ChangePointDetecion

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    cfg['seed'] = cfg['init_seed']
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    pivot_data_name = 'EXP'
    num_dims = 4
    data_names = ['{}-{}'.format(pivot_data_name, x + 1) for x in range(num_dims)]
    test_modes = ['cusum', 'scusum', 'scanb']
    runtime = defaultdict(list)
    cfg['control']['change'] = '1.0'
    for data_name in data_names:
        print(data_name)
        cfg['control']['data_name'] = data_name
        process_control()
        cfg['num_trials'] = 1
        cfg['params'] = make_params(cfg['data_name'])
        cfg['footprint'] = make_footprint(cfg['params'])
        dataset = fetch_dataset(cfg['data_name'], cfg['params'])
        process_dataset(dataset)
        data_loader = make_data_loader(dataset, 'cpd')
        for test_mode in test_modes:
            print(test_mode)
            cfg['test_mode'] = test_mode
            s = time.time()
            cpd = ChangePointDetecion(cfg['test_mode'], cfg['arl'], cfg['noise'], dataset['test'])
            test(data_loader['test'], cpd)
            e = time.time()
            runtime[test_mode].append(e - s)
    print(runtime)
    result = {'runtime': runtime}
    save(result, os.path.join('output', 'result', '{}_runtime.pt'.format(cfg['seed'])))
    return


def test(data_loader, cpd):
    for i, input in enumerate(data_loader):
        input = collate(input)
        input = to_device(input, cfg['device'])
        output = cpd.test(input)
    return


if __name__ == "__main__":
    main()
