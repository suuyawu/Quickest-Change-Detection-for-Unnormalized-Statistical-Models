import argparse
import datetime
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg, process_args
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, resume, collate
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
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    process_control()
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'], cfg['params'])
    process_dataset(dataset)
    metric = Metric({'test': ['EDD']})
    data_loader = make_data_loader(dataset, 'cpd')
    cpd = ChangePointDetecion(cfg['test_mode'], cfg['arl'], cfg['noise'], dataset['test'])
    logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    test(data_loader['test'], cpd, metric, logger)
    cpd.clean()
    result = {'cfg': cfg, 'logger': logger, 'cpd': cpd}
    save(result, os.path.join('output', 'result', '{}.pt'.format(cfg['model_tag'])))
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
