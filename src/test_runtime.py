import argparse
import datetime
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import models
from collections import defaultdict
from config import cfg, process_args
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, resume, collate, make_params, make_footprint
from logger import make_logger
from modules import ChangePointDetecion

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    cfg['seed'] = 0
    process_control()
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    pivot_data_name = 'EXP'
    num_dims = 3
    data_names = ['{}-{}'.format(pivot_data_name, x) for x in range(num_dims)]
    test_modes = ['cusum', 'scusum', 'scanb']
    runtime = defaultdict(list)
    cfg['num_trials'] = 1
    for data_name in data_names:
        cfg['data_name'] = data_name
        cfg['params'] = make_params(cfg['data_name'])
        cfg['footprint'] = make_footprint(cfg['params'])
        dataset = fetch_dataset(cfg['data_name'], cfg['params'])
        process_dataset(dataset)
        metric = Metric({'test': ['EDD']})
        logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
        data_loader = make_data_loader(dataset, 'cpd')
        for test_mode in test_modes:
            cfg['test_mode'] = test_mode
            s = time.time()
            cpd = ChangePointDetecion(cfg['test_mode'], cfg['arl'], cfg['noise'], dataset['test'])
            test(data_loader['test'], cpd, metric, logger)
            e = time.time()
            runtime[test_mode].append(s - e)
    result = {'runtime': runtime}
    save(result, os.path.join('output', 'result', 'runtime.pt'))
    return


def test(data_loader, cpd, metric, logger):
    start_time = time.time()
    for i, input in enumerate(data_loader):
        logger.save(True)
        input = collate(input)
        input = to_device(input, cfg['device'])
        output = cpd.test(input)
        evaluation = metric.evaluate(metric.metric_name['test'], input, output)
        logger.append(evaluation, 'test', 1)
        if i % np.ceil((len(data_loader) * cfg['log_interval'])) == 0:
            batch_time = (time.time() - start_time) / (i + 1)
            exp_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader) - i - 1)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Test Iter: {}/{}({:.0f}%)'.format(i + 1, len(data_loader), 100. * i / len(data_loader)),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'test', mean=False)
            print(logger.write('test', metric.metric_name['test']))
        logger.save(False)
    info = {'info': ['Model: {}'.format(cfg['model_tag']),
                     'Test Iter: {}/{}(100%)'.format(len(data_loader), len(data_loader))]}
    logger.append(info, 'test', mean=False)
    print(logger.write('test', metric.metric_name['test']))
    return


if __name__ == "__main__":
    main()
