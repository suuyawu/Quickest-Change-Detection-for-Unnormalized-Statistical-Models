import argparse
import itertools

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--init_gpu', default=0, type=int)
parser.add_argument('--num_gpus', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--init_seed', default=0, type=int)
parser.add_argument('--round', default=4, type=int)
parser.add_argument('--experiment_step', default=1, type=int)
parser.add_argument('--num_experiments', default=1, type=int)
parser.add_argument('--resume_mode', default=0, type=int)
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--split_round', default=65535, type=int)
args = vars(parser.parse_args())


def make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    control_names = [control_names]
    controls = script_name + init_seeds + world_size + num_experiments + resume_mode + control_names
    controls = list(itertools.product(*controls))
    return controls


def main():
    run = args['run']
    init_gpu = args['init_gpu']
    num_gpus = args['num_gpus']
    world_size = args['world_size']
    round = args['round']
    experiment_step = args['experiment_step']
    init_seed = args['init_seed']
    num_experiments = args['num_experiments']
    resume_mode = args['resume_mode']
    mode = args['mode']
    split_round = args['split_round']
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in
               list(range(init_gpu, init_gpu + num_gpus, world_size))]
    init_seeds = [list(range(init_seed, init_seed + num_experiments, experiment_step))]
    world_size = [[world_size]]
    num_experiments = [[experiment_step]]
    resume_mode = [[resume_mode]]
    filename = '{}_{}'.format(run, mode)
    if mode == 'data':
        script_name = [['make_datasets.py']]
        data_names = ['MVN-2']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        change_mean = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
                       0.9, 0.95, 1]
        change_logvar = float(0)
        for i in range(len(change_mean)):
            change_mean_i = float(change_mean[i])
            change_i = '{}-{}'.format(change_mean_i, change_logvar)
            change.append(change_i)
        noise = ['0']
        test_mode = ['scusum']
        arl = ['2000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        mvn_mean_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                          control_name)
        data_names = ['MVN-2']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        change_mean = float(0)
        change_logvar = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
                         0.85, 0.9, 0.95, 1]
        for i in range(len(change_logvar)):
            change_logvar_i = float(change_logvar[i])
            change_i = '{}-{}'.format(change_mean, change_logvar_i)
            change.append(change_i)
        noise = ['0']
        test_mode = ['scusum']
        arl = ['2000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        mvn_logvar_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                            control_name)
        data_names = ['EXP-2']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        change_tau = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                      2.0]
        for i in range(len(change_tau)):
            change_tau_i = float(change_tau[i])
            change_i = '{}'.format(change_tau_i)
            change.append(change_i)
        noise = ['0']
        test_mode = ['scusum']
        arl = ['2000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        exp_tau_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                         control_name)
        data_names = ['RBM-50']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        change_W = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07,
                    0.075, 0.08, 0.085, 0.09, 0.095, 0.1]
        for i in range(len(change_W)):
            change_W_i = float(change_W[i])
            change_i = '{}'.format(change_W_i)
            change.append(change_i)
        noise = ['0']
        test_mode = ['scusum']
        arl = ['2000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        rbm_W_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                       control_name)
        controls = mvn_mean_controls + mvn_logvar_controls + exp_tau_controls + rbm_W_controls
    elif mode == 'mvn-mean':
        script_name = [['{}_cpd.py'.format(run)]]
        data_names = ['MVN-2']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        change_mean = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
                       0.9, 0.95, 1]
        change_logvar = float(0)
        for i in range(len(change_mean)):
            change_mean_i = float(change_mean[i])
            change_i = '{}-{}'.format(change_mean_i, change_logvar)
            change.append(change_i)
        noise = ['0']
        test_mode = ['cusum', 'scusum', 'scanb', 'calm']
        arl = ['2000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    elif mode == 'mvn-mean-arl':
        script_name = [['{}_cpd.py'.format(run)]]
        data_names = ['MVN-2']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        change_mean = [0, 0.5]
        change_logvar = float(0)
        for i in range(len(change_mean)):
            change_mean_i = float(change_mean[i])
            change_i = '{}-{}'.format(change_mean_i, change_logvar)
            change.append(change_i)
        noise = ['0']
        test_mode = ['cusum', 'scusum', 'scanb', 'calm']
        arl = ['500', '1000', '1500', '2500', '5000', '7500', '10000', '15000', '20000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    elif mode == 'mvn-mean-noise':
        script_name = [['{}_cpd.py'.format(run)]]
        data_names = ['MVN-2']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        change_mean = [0, 0.5]
        change_logvar = float(0)
        for i in range(len(change_mean)):
            change_mean_i = float(change_mean[i])
            change_i = '{}-{}'.format(change_mean_i, change_logvar)
            change.append(change_i)
        noise = ['0.005', '0.01', '0.05', '0.1', '0.5', '1']
        test_mode = ['cusum', 'scusum', 'scanb', 'calm']
        arl = ['2000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    elif mode == 'mvn-logvar':
        script_name = [['{}_cpd.py'.format(run)]]
        data_names = ['MVN-2']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        change_mean = float(0)
        change_logvar = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
                         0.85, 0.9, 0.95, 1]
        for i in range(len(change_logvar)):
            change_logvar_i = float(change_logvar[i])
            change_i = '{}-{}'.format(change_mean, change_logvar_i)
            change.append(change_i)
        noise = ['0']
        test_mode = ['cusum', 'scusum', 'scanb', 'calm']
        arl = ['2000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    elif mode == 'mvn-logvar-arl':
        script_name = [['{}_cpd.py'.format(run)]]
        data_names = ['MVN-2']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        change_mean = float(0)
        change_logvar = [0, 0.5]
        for i in range(len(change_logvar)):
            change_logvar_i = float(change_logvar[i])
            change_i = '{}-{}'.format(change_mean, change_logvar_i)
            change.append(change_i)
        noise = ['0']
        test_mode = ['cusum', 'scusum', 'scanb', 'calm']
        arl = ['500', '1000', '1500', '2500', '5000', '7500', '10000', '15000', '20000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    elif mode == 'mvn-logvar-noise':
        script_name = [['{}_cpd.py'.format(run)]]
        data_names = ['MVN-2']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        change_mean = float(0)
        change_logvar = [0, 0.5]
        for i in range(len(change_logvar)):
            change_logvar_i = float(change_logvar[i])
            change_i = '{}-{}'.format(change_mean, change_logvar_i)
            change.append(change_i)
        noise = ['0.005', '0.01', '0.05', '0.1', '0.5', '1']
        test_mode = ['cusum', 'scusum', 'scanb', 'calm']
        arl = ['2000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    elif mode == 'exp-tau':
        script_name = [['{}_cpd.py'.format(run)]]
        data_names = ['EXP-2']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        change_tau = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                      2.0]
        for i in range(len(change_tau)):
            change_tau_i = float(change_tau[i])
            change_i = '{}'.format(change_tau_i)
            change.append(change_i)
        noise = ['0']
        test_mode = ['cusum', 'scusum', 'scanb', 'calm']
        arl = ['2000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    elif mode == 'exp-tau-arl':
        script_name = [['{}_cpd.py'.format(run)]]
        data_names = ['EXP-2']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        change_tau = [0, 1.0]
        for i in range(len(change_tau)):
            change_tau_i = float(change_tau[i])
            change_i = '{}'.format(change_tau_i)
            change.append(change_i)
        noise = ['0']
        test_mode = ['cusum', 'scusum', 'scanb', 'calm']
        arl = ['500', '1000', '1500', '2500', '5000', '7500', '10000', '15000', '20000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    elif mode == 'exp-tau-noise':
        script_name = [['{}_cpd.py'.format(run)]]
        data_names = ['EXP-2']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        change_tau = [0, 1.0]
        for i in range(len(change_tau)):
            change_tau_i = float(change_tau[i])
            change_i = '{}'.format(change_tau_i)
            change.append(change_i)
        noise = ['0.005', '0.01', '0.05', '0.1', '0.5', '1']
        test_mode = ['cusum', 'scusum', 'scanb', 'calm']
        arl = ['2000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    elif mode == 'rbm-W':
        script_name = [['{}_cpd.py'.format(run)]]
        data_names = ['RBM-50']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        change_W = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07,
                    0.075, 0.08, 0.085, 0.09, 0.095, 0.1]
        for i in range(len(change_W)):
            change_W_i = float(change_W[i])
            change_i = '{}'.format(change_W_i)
            change.append(change_i)
        noise = ['0']
        test_mode = ['scusum', 'scanb', 'calm']
        arl = ['2000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    elif mode == 'rbm-W-arl':
        script_name = [['{}_cpd.py'.format(run)]]
        data_names = ['RBM-50']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        change_W = [0, 0.05]
        for i in range(len(change_W)):
            change_W_i = float(change_W[i])
            change_i = '{}'.format(change_W_i)
            change.append(change_i)
        noise = ['0']
        test_mode = ['scusum', 'scanb', 'calm']
        arl = ['500', '1000', '1500', '2500', '5000', '7500', '10000', '15000', '20000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    elif mode == 'rbm-W-noise':
        script_name = [['{}_cpd.py'.format(run)]]
        data_names = ['RBM-50']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        change_W = [0, 0.05]
        for i in range(len(change_W)):
            change_W_i = float(change_W[i])
            change_i = '{}'.format(change_W_i)
            change.append(change_i)
        noise = ['0.005', '0.01', '0.05', '0.1', '0.5', '1']
        test_mode = ['scusum', 'scanb', 'calm']
        arl = ['2000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    else:
        raise ValueError('Not valid mode')
    s = '#!/bin/bash\n'
    j = 1
    k = 1
    for i in range(len(controls)):
        controls[i] = list(controls[i])
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} --num_experiments {} ' \
                '--resume_mode {} --control_name {}&\n'.format(gpu_ids[i % len(gpu_ids)], *controls[i])
        if i % round == round - 1:
            s = s[:-2] + '\nwait\n'
            if j % split_round == 0:
                print(s)
                run_file = open('{}_{}.sh'.format(filename, k), 'w')
                run_file.write(s)
                run_file.close()
                s = '#!/bin/bash\n'
                k = k + 1
            j = j + 1
    if s != '#!/bin/bash\n':
        if s[-5:-1] != 'wait':
            s = s + 'wait\n'
        print(s)
        run_file = open('{}_{}.sh'.format(filename, k), 'w')
        run_file.write(s)
        run_file.close()
    return


if __name__ == '__main__':
    main()
