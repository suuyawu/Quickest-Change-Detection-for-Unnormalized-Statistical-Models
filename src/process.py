import os
import itertools
import numpy as np
import pandas as pd
from utils import save, load, makedir_exist_ok
import matplotlib.pyplot as plt
from collections import defaultdict
from brokenaxes import brokenaxes

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
result_path = os.path.join('output', 'result')
save_format = 'pdf'
vis_path = os.path.join('output', 'vis', '{}'.format(save_format))
num_experiments = 1
exp = [str(x) for x in list(range(num_experiments))]
dpi = 300
write = {'mean': True, 'history': False}


def make_controls(control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    controls = [exp] + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_all_controls(mode):
    if mode == 'mvn-mean':
        data_names = ['MVN-2']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        # change_mean = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
        #                0.9, 0.95, 1]
        change_mean = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        change_logvar = float(0)
        for i in range(len(change_mean)):
            change_mean_i = float(change_mean[i])
            change_i = '{}-{}'.format(change_mean_i, change_logvar)
            change.append(change_i)
        noise = ['0']
        test_mode = ['cusum', 'scusum', 'scanb', 'calm']
        arl = ['2000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        controls = make_controls(control_name)
    elif mode == 'mvn-mean-arl':
        data_names = ['MVN-2']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        change_mean = [0, 0.1]
        change_logvar = float(0)
        for i in range(len(change_mean)):
            change_mean_i = float(change_mean[i])
            change_i = '{}-{}'.format(change_mean_i, change_logvar)
            change.append(change_i)
        noise = ['0']
        test_mode = ['cusum', 'scusum', 'scanb', 'calm']
        arl = ['500', '1000', '1500', '2500', '5000', '7500', '10000', '15000', '20000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        controls = make_controls(control_name)
    elif mode == 'mvn-mean-lambda':
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
        test_mode = ['scusum']
        arl = ['10000']
        pre_length = ['10', '20', '30', '40', '50', '100', '200', '300', '400', '500']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl, pre_length]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    elif mode == 'mvn-mean-noise':
        data_names = ['MVN-2']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        change_mean = [0, 0.1]
        change_logvar = float(0)
        for i in range(len(change_mean)):
            change_mean_i = float(change_mean[i])
            change_i = '{}-{}'.format(change_mean_i, change_logvar)
            change.append(change_i)
        noise = ['0.005', '0.01', '0.05', '0.1', '0.5', '1']
        test_mode = ['cusum', 'scusum', 'scanb', 'calm']
        arl = ['2000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        controls = make_controls(control_name)
    elif mode == 'mvn-logvar':
        data_names = ['MVN-2']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        change_mean = float(0)
        # change_logvar = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
        #                  0.85, 0.9, 0.95, 1]
        change_logvar = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        for i in range(len(change_logvar)):
            change_logvar_i = float(change_logvar[i])
            change_i = '{}-{}'.format(change_mean, change_logvar_i)
            change.append(change_i)
        noise = ['0']
        test_mode = ['cusum', 'scusum', 'scanb', 'calm']
        arl = ['2000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        controls = make_controls(control_name)
    elif mode == 'mvn-logvar-arl':
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
        controls = make_controls(control_name)
    elif mode == 'mvn-logvar-noise':
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
        controls = make_controls(control_name)
    elif mode == 'exp-tau':
        data_names = ['EXP-2']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        # change_tau = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
        #               2.0]
        change_tau = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i in range(len(change_tau)):
            change_tau_i = float(change_tau[i])
            change_i = '{}'.format(change_tau_i)
            change.append(change_i)
        noise = ['0']
        test_mode = ['cusum', 'scusum', 'scanb', 'calm']
        arl = ['2000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        controls = make_controls(control_name)
    elif mode == 'exp-tau-arl':
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
        controls = make_controls(control_name)
    elif mode == 'exp-tau-noise':
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
        controls = make_controls(control_name)
    elif mode == 'rbm-W':
        data_names = ['RBM-50']
        num_pre = ['500']
        num_post = ['10000']
        change = []
        # change_W = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07,
        #             0.075, 0.08, 0.085, 0.09, 0.095, 0.1]
        change_W = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
        for i in range(len(change_W)):
            change_W_i = float(change_W[i])
            change_i = '{}'.format(change_W_i)
            change.append(change_i)
        noise = ['0']
        test_mode = ['scusum', 'scanb', 'calm']
        arl = ['2000']
        control_name = [[data_names, num_pre, num_post, change, noise, test_mode, arl]]
        controls = make_controls(control_name)
    elif mode == 'rbm-W-arl':
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
        controls = make_controls(control_name)
    elif mode == 'rbm-W-noise':
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
        controls = make_controls(control_name)
    else:
        raise ValueError('Not valid mode')
    return controls


def main():
    modes = ['mvn-mean', 'mvn-mean-arl', 'mvn-mean-noise',
             'mvn-logvar', 'mvn-logvar-arl', 'mvn-logvar-noise',
             'exp-tau', 'exp-tau-arl', 'exp-tau-noise',
             'rbm-W', 'rbm-W-arl', 'rbm-W-noise']
    # modes = ['mvn-mean', 'mvn-mean-arl', 'mvn-mean-noise']
    # modes = ['mvn-mean', 'mvn-logvar']
    controls = []
    for mode in modes:
        controls += make_all_controls(mode)
    processed_result = process_result(controls)
    df_mean = make_df(processed_result, 'mean')
    df_history = make_df(processed_result, 'history')
    make_vis_runtime()
    make_vis_score(df_history)
    make_vis_change(df_mean)
    make_vis_arl(df_mean)
    make_vis_noise(df_mean)
    return


def tree():
    return defaultdict(tree)


def process_result(controls):
    result = tree()
    for control in controls:
        model_tag = '_'.join(control)
        gather_result(list(control), model_tag, result)
    processed_result = tree()
    extract_result(processed_result, result, [])
    return processed_result


def gather_result(control, model_tag, processed_result):
    if len(control) == 1:
        base_result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            for metric_name in base_result['logger'].history:
                processed_result[metric_name]['mean']['summary']['mean'] = np.mean(
                    base_result['logger'].history[metric_name])
                processed_result[metric_name]['mean']['summary']['std'] = np.std(
                    base_result['logger'].history[metric_name])
            score = np.array(base_result['cpd'].stats['score']).reshape((100, -1))
            detect = np.array(base_result['cpd'].stats['detect']).reshape((100, -1))
            threshold = np.array(base_result['cpd'].stats['threshold']).reshape((100, -1))
            mask = np.any((score == float('inf')) | np.isnan(score), axis=0)
            score = score[:, ~mask]
            detect = detect[:, ~mask]
            if score.shape[1] > 0:
                processed_result['test/score']['history']['summary']['mean'] = score.mean(axis=0)
                processed_result['test/score']['history']['summary']['std'] = score.std(axis=0)
                processed_result['test/detect']['mean']['summary']['mean'] = detect.mean()
                processed_result['test/detect']['mean']['summary']['std'] = detect.std()
                processed_result['test/threshold']['history']['summary']['mean'] = threshold.mean(axis=0)
                processed_result['test/threshold']['history']['summary']['std'] = threshold.std(axis=0)
            else:
                print(model_tag)
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        gather_result([control[0]] + control[2:], model_tag, processed_result[control[1]])
    return


def extract_result(extracted_processed_result, processed_result, control):
    def extract(metric_name, mode):
        output = False
        if metric_name in ['test/CADD', 'test/detect']:
            if mode == 'mean':
                output = True
        elif metric_name in ['test/score', 'test/threshold']:
            if mode == 'history':
                output = True
        return output

    if 'summary' in processed_result:
        control_name, metric_name, mode = control
        if not extract(metric_name, mode):
            return
        stats = ['mean', 'std']
        for stat in stats:
            exp_name = '_'.join([control_name, metric_name.split('/')[1], stat])
            extracted_processed_result[mode][exp_name] = processed_result['summary'][stat]
    else:
        for k, v in processed_result.items():
            extract_result(extracted_processed_result, v, control + [k])
    return


def make_df(processed_result, mode):
    df = defaultdict(list)
    for exp_name in processed_result[mode]:
        exp_name_list = exp_name.split('_')
        df_name = '_'.join([*exp_name_list])
        index_name = [1]
        df[df_name].append(pd.DataFrame(data=processed_result[mode][exp_name].reshape(1, -1), index=index_name))
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
    if write[mode]:
        startrow = 0
        writer = pd.ExcelWriter('{}/result_{}.xlsx'.format(result_path, mode), engine='xlsxwriter')
        for df_name in df:
            df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
            writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
            startrow = startrow + len(df[df_name].index) + 3
        writer.save()
    return df


def make_vis_runtime():
    label_dict = {'cusum': 'CUSUM', 'scusum': 'SCUSUM', 'scanb': 'Scan B-statistic', 'calm': 'CALM-MMD'}
    color_dict = {'cusum': 'black', 'scusum': 'red', 'scanb': 'blue', 'calm': 'cyan'}
    linestyle_dict = {'cusum': '-', 'scusum': '--', 'scanb': ':', 'calm': (0, (5, 5))}
    marker_dict = {'cusum': 'D', 'scusum': 'o', 'scanb': 'p', 'calm': 's'}
    fontsize_dict = {'legend': 16, 'label': 16, 'ticks': 16}
    loc_dict = {'runtime': 'upper left'}
    figsize = (5, 4)
    num_experiments = 1
    num_dims = 4
    num_dims = list(range(1, num_dims + 1))
    runtime = []
    for i in range(num_experiments):
        runtime_i = load(os.path.join('output', 'result', '{}_runtime.pt'.format(i)))['runtime']
        runtime.append(runtime_i)
    fig = plt.figure(figsize=figsize)
    ax_1 = fig.add_subplot(111)
    for label in runtime[0]:
        runtime_ = np.array([runtime[i][label] for i in range(num_experiments)])
        runtime_mean = runtime_.mean(axis=0)
        ax_1.plot(num_dims, runtime_mean, color=color_dict[label],
                  linestyle=linestyle_dict[label], label=label_dict[label], marker=marker_dict[label])
    ax_1.set_xlabel('Dimension', fontsize=fontsize_dict['label'])
    ax_1.set_ylabel('CPU Time (s)', fontsize=fontsize_dict['label'])
    ax_1.set_xticks(num_dims)
    ax_1.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
    ax_1.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
    ax_1.legend(loc=loc_dict['runtime'], fontsize=fontsize_dict['legend'])
    ax_1.grid(linestyle='--', linewidth='0.5')
    ax_1.set_yscale('log')
    plt.tight_layout()
    dir_name = 'time'
    dir_path = os.path.join(vis_path, dir_name)
    fig_path = os.path.join(dir_path, '{}.{}'.format('runtime', save_format))
    makedir_exist_ok(dir_path)
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    return


def make_vis_score(df_history):
    label_dict = {'cusum': 'CUSUM', 'scusum': 'SCUSUM', 'scanb': 'Scan B-statistic', 'calm': 'CALM-MMD',
                  'threshold': 'Threshold', 'cp': 'Change Point'}
    color_dict = {'cusum': 'black', 'scusum': 'red', 'scanb': 'blue', 'calm': 'cyan',
                  'threshold': 'orange', 'cp': 'green'}
    linestyle_dict = {'cusum': '-', 'scusum': '--', 'scanb': ':', 'calm': '-.', 'threshold': (0, (5, 5)),
                      'cp': (0, (10, 5))}
    marker_dict = {'cusum': 'D', 'scusum': 'o', 'scanb': 'p', 'calm': 's'}
    fontsize_dict = {'legend': 16, 'label': 16, 'ticks': 16}
    loc_dict = {'score': 'upper right'}
    fig = {}
    ax_dict_1 = {}
    ax_dict_2 = {}
    ax_dict_3 = {}
    ax_dict_4 = {}
    for df_name in df_history:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        noise = df_name_list[4]
        test_mode = df_name_list[5]
        arl = df_name_list[6]
        mask = noise == '0' and test_mode == 'scusum' and arl == '2000' and metric_name == 'score' and stat == 'mean'
        if mask:
            data_name = df_name_list[0]
            fig_name = '_'.join([*df_name_list])
            if 'RBM' not in data_name:
                figsize = (20, 4)
                fig[fig_name] = plt.figure(fig_name, figsize=figsize)
                if fig_name not in ax_dict_1:
                    ax_dict_1[fig_name] = fig[fig_name].add_subplot(141)
                    ax_dict_2[fig_name] = fig[fig_name].add_subplot(142)
                    ax_dict_3[fig_name] = fig[fig_name].add_subplot(143)
                    ax_dict_4[fig_name] = fig[fig_name].add_subplot(144)

                df_name_cusum = '_'.join([*df_name_list[:5], 'cusum', *df_name_list[6:-1], stat])
                df_name_cusum_std = '_'.join([*df_name_list[:5], 'cusum', *df_name_list[6:-1], 'std'])
                df_name_scusum = '_'.join([*df_name_list[:-1], stat])
                df_name_scusum_std = '_'.join([*df_name_list[:-1], 'std'])
                df_name_scanb = '_'.join([*df_name_list[:5], 'scanb', *df_name_list[6:-1], stat])
                df_name_scanb_std = '_'.join([*df_name_list[:5], 'scanb', *df_name_list[6:-1], 'std'])
                df_name_calm = '_'.join([*df_name_list[:5], 'calm', *df_name_list[6:-1], stat])
                df_name_calm_std = '_'.join([*df_name_list[:5], 'calm', *df_name_list[6:-1], 'std'])

                cusum = df_history[df_name_cusum].iloc[0].to_numpy()[:2000]
                cusum_std = df_history[df_name_cusum_std].iloc[0].to_numpy()[:2000]
                scusum = df_history[df_name_scusum].iloc[0].to_numpy()[:2000]
                scusum_std = df_history[df_name_scusum_std].iloc[0].to_numpy()[:2000]
                scanb = df_history[df_name_scanb].iloc[0].to_numpy()[:2000]
                scanb_std = df_history[df_name_scanb_std].iloc[0].to_numpy()[:2000]
                calm = df_history[df_name_calm].iloc[0].to_numpy()[:2000]
                calm_std = df_history[df_name_calm_std].iloc[0].to_numpy()[:2000]

                df_name_cusum_threshold = '_'.join([*df_name_list[:5], 'cusum', *df_name_list[6:-2],
                                                    'threshold', stat])
                df_name_cusum_threshold_std = '_'.join([*df_name_list[:5], 'cusum', *df_name_list[6:-2],
                                                        'threshold', 'std'])
                df_name_scusum_threshold = '_'.join([*df_name_list[:-2], 'threshold', stat])
                df_name_scusum_threshold_std = '_'.join([*df_name_list[:-2], 'threshold', 'std'])
                df_name_scanb_threshold = '_'.join([*df_name_list[:5], 'scanb', *df_name_list[6:-2],
                                                    'threshold', stat])
                df_name_scanb_threshold_std = '_'.join([*df_name_list[:5], 'scanb', *df_name_list[6:-2],
                                                        'threshold', 'std'])
                df_name_calm_threshold = '_'.join([*df_name_list[:5], 'calm', *df_name_list[6:-2],
                                                   'threshold', stat])
                df_name_calm_threshold_std = '_'.join([*df_name_list[:5], 'calm', *df_name_list[6:-2],
                                                       'threshold', 'std'])

                cusum_threshold = df_history[df_name_cusum_threshold].iloc[0].to_numpy()[:2000]
                cusum_threshold_std = df_history[df_name_cusum_threshold_std].iloc[0].to_numpy()[:2000]
                scusum_threshold = df_history[df_name_scusum_threshold].iloc[0].to_numpy()[:2000]
                scusum_threshold_std = df_history[df_name_scusum_threshold_std].iloc[0].to_numpy()[:2000]
                scanb_threshold = df_history[df_name_scanb_threshold].iloc[0].to_numpy()[:2000]
                scanb_threshold_std = df_history[df_name_scanb_threshold_std].iloc[0].to_numpy()[:2000]
                calm_threshold = df_history[df_name_calm_threshold].iloc[0].to_numpy()[:2000]
                calm_threshold_std = df_history[df_name_calm_threshold_std].iloc[0].to_numpy()[:2000]

                x = range(1, len(cusum) + 1)
                ax_1 = ax_dict_1[fig_name]
                xlabel = 'Time Step'
                ylabel = 'Detection Score'
                pivot = 'cusum'
                ax_1.plot(x, cusum, label=label_dict[pivot], color=color_dict[pivot],
                          linestyle=linestyle_dict[pivot])
                ax_1.fill_between(x, (cusum - cusum_std / 10), (cusum + cusum_std / 10), color=color_dict[pivot],
                                  alpha=.1)
                ax_1.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
                ax_1.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
                ax_1.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
                ax_1.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
                ylim_1 = list(ax_1.get_ylim())
                ylim_1[1] = max(ylim_1[1], max(cusum_threshold))
                ax_1.plot(x, cusum_threshold, label=label_dict['threshold'], color=color_dict['threshold'],
                          linestyle=linestyle_dict['threshold'])
                ax_1.fill_between(x, (cusum_threshold - cusum_threshold_std / 10),
                                  (cusum_threshold + cusum_threshold_std / 10), color=color_dict['threshold'], alpha=.1)
                ax_1.vlines(500, ylim_1[0], ylim_1[1], label=label_dict['cp'], color=color_dict['cp'],
                            linestyle=linestyle_dict['cp'])
                ax_1.legend(loc=loc_dict['score'], fontsize=fontsize_dict['legend'])

                x = range(1, len(scusum) + 1)
                ax_2 = ax_dict_2[fig_name]
                xlabel = 'Time Step'
                ylabel = 'Detection Score'
                pivot = 'scusum'
                ax_2.plot(x, scusum, label=label_dict[pivot], color=color_dict[pivot],
                          linestyle=linestyle_dict[pivot])
                ax_2.fill_between(x, (scusum - scusum_std / 10), (scusum + scusum_std / 10), color=color_dict[pivot],
                                  alpha=.1)
                ax_2.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
                ax_2.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
                ax_2.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
                ax_2.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
                ylim_2 = list(ax_2.get_ylim())
                ylim_2[1] = max(ylim_2[1], max(scusum_threshold))
                ax_2.plot(x, scusum_threshold, label=label_dict['threshold'], color=color_dict['threshold'],
                          linestyle=linestyle_dict['threshold'])
                ax_2.fill_between(x, (scusum_threshold - scusum_threshold_std / 10),
                                  (scusum_threshold + scusum_threshold_std / 10),
                                  color=color_dict['threshold'], alpha=.1)
                ax_2.vlines(500, ylim_2[0], ylim_2[1], label=label_dict['cp'], color=color_dict['cp'],
                            linestyle=linestyle_dict['cp'])
                ax_2.legend(loc=loc_dict['score'], fontsize=fontsize_dict['legend'])

                x = range(1, len(scanb) + 1)
                ax_3 = ax_dict_3[fig_name]
                xlabel = 'Time Step'
                ylabel = 'Detection Score'
                pivot = 'scanb'
                ax_3.plot(x, scanb, label=label_dict[pivot], color=color_dict[pivot],
                          linestyle=linestyle_dict[pivot])
                ax_3.fill_between(x, (scanb - scanb_std / 10), (scanb + scanb_std / 10), color=color_dict[pivot],
                                  alpha=.1)
                ax_3.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
                ax_3.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
                ax_3.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
                ax_3.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
                ylim_3 = list(ax_3.get_ylim())
                ylim_3[1] = max(ylim_3[1], max(scanb_threshold))
                ax_3.plot(x, scanb_threshold, label=label_dict['threshold'], color=color_dict['threshold'],
                          linestyle=linestyle_dict['threshold'])
                ax_3.fill_between(x, (scanb_threshold - scanb_threshold_std / 10),
                                  (scanb_threshold + scanb_threshold_std / 10), color=color_dict['threshold'], alpha=.1)
                ax_3.vlines(500, ylim_3[0], ylim_3[1], label=label_dict['cp'], color=color_dict['cp'],
                            linestyle=linestyle_dict['cp'])
                ax_3.legend(loc=loc_dict['score'], fontsize=fontsize_dict['legend'])

                x = range(1, len(calm) + 1)
                ax_4 = ax_dict_4[fig_name]
                xlabel = 'Time Step'
                ylabel = 'Detection Score'
                pivot = 'calm'
                ax_4.plot(x, calm, label=label_dict[pivot], color=color_dict[pivot],
                          linestyle=linestyle_dict[pivot])
                ax_4.fill_between(x, (calm - calm_std / 10), (calm + calm_std / 10), color=color_dict[pivot],
                                  alpha=.1)
                ax_4.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
                ax_4.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
                ax_4.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
                ax_4.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
                ylim_4 = list(ax_4.get_ylim())
                ylim_4[1] = max(ylim_4[1], max(calm_threshold))
                ax_4.plot(x, calm_threshold, label=label_dict['threshold'], color=color_dict['threshold'],
                          linestyle=linestyle_dict['threshold'])
                ax_4.fill_between(x, (calm_threshold - calm_threshold_std / 10),
                                  (calm_threshold + calm_threshold_std / 10), color=color_dict['threshold'], alpha=.1)
                ax_4.vlines(500, ylim_4[0], ylim_4[1], label=label_dict['cp'], color=color_dict['cp'],
                            linestyle=linestyle_dict['cp'])
                ax_4.legend(loc=loc_dict['score'], fontsize=fontsize_dict['legend'])
            else:
                figsize = (15, 4)
                fig[fig_name] = plt.figure(fig_name, figsize=figsize)
                if fig_name not in ax_dict_1:
                    ax_dict_1[fig_name] = fig[fig_name].add_subplot(131)
                    ax_dict_2[fig_name] = fig[fig_name].add_subplot(132)
                    ax_dict_3[fig_name] = fig[fig_name].add_subplot(133)

                df_name_scusum = '_'.join([*df_name_list[:-1], stat])
                df_name_scusum_std = '_'.join([*df_name_list[:-1], 'std'])
                df_name_scanb = '_'.join([*df_name_list[:5], 'scanb', *df_name_list[6:-1], stat])
                df_name_scanb_std = '_'.join([*df_name_list[:5], 'scanb', *df_name_list[6:-1], 'std'])
                df_name_calm = '_'.join([*df_name_list[:5], 'calm', *df_name_list[6:-1], stat])
                df_name_calm_std = '_'.join([*df_name_list[:5], 'calm', *df_name_list[6:-1], 'std'])

                scusum = df_history[df_name_scusum].iloc[0].to_numpy()[:2000]
                scusum_std = df_history[df_name_scusum_std].iloc[0].to_numpy()[:2000]
                scanb = df_history[df_name_scanb].iloc[0].to_numpy()[:2000]
                scanb_std = df_history[df_name_scanb_std].iloc[0].to_numpy()[:2000]
                calm = df_history[df_name_calm].iloc[0].to_numpy()[:2000]
                calm_std = df_history[df_name_calm_std].iloc[0].to_numpy()[:2000]

                df_name_scusum_threshold = '_'.join([*df_name_list[:-2], 'threshold', stat])
                df_name_scusum_threshold_std = '_'.join([*df_name_list[:-2], 'threshold', 'std'])
                df_name_scanb_threshold = '_'.join([*df_name_list[:5], 'scanb', *df_name_list[6:-2],
                                                    'threshold', stat])
                df_name_scanb_threshold_std = '_'.join([*df_name_list[:5], 'scanb', *df_name_list[6:-2],
                                                        'threshold', 'std'])
                df_name_calm_threshold = '_'.join([*df_name_list[:5], 'calm', *df_name_list[6:-2],
                                                   'threshold', stat])
                df_name_calm_threshold_std = '_'.join([*df_name_list[:5], 'calm', *df_name_list[6:-2],
                                                       'threshold', 'std'])

                scusum_threshold = df_history[df_name_scusum_threshold].iloc[0].to_numpy()[:2000]
                scusum_threshold_std = df_history[df_name_scusum_threshold_std].iloc[0].to_numpy()[:2000]
                scanb_threshold = df_history[df_name_scanb_threshold].iloc[0].to_numpy()[:2000]
                scanb_threshold_std = df_history[df_name_scanb_threshold_std].iloc[0].to_numpy()[:2000]
                calm_threshold = df_history[df_name_calm_threshold].iloc[0].to_numpy()[:2000]
                calm_threshold_std = df_history[df_name_calm_threshold_std].iloc[0].to_numpy()[:2000]

                x = range(1, len(scusum) + 1)
                ax_1 = ax_dict_1[fig_name]
                xlabel = 'Time Step'
                ylabel = 'Detection Score'
                pivot = 'scusum'
                ax_1.plot(x, scusum, label=label_dict[pivot], color=color_dict[pivot],
                          linestyle=linestyle_dict[pivot])
                ax_1.fill_between(x, (scusum - scusum_std / 10), (scusum + scusum_std / 10), color=color_dict[pivot],
                                  alpha=.1)
                ax_1.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
                ax_1.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
                ax_1.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
                ax_1.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
                ylim_1 = list(ax_1.get_ylim())
                ylim_1[1] = max(ylim_1[1], max(scusum_threshold))
                ax_1.plot(x, scusum_threshold, label=label_dict['threshold'], color=color_dict['threshold'],
                          linestyle=linestyle_dict['threshold'])
                ax_1.fill_between(x, (scusum_threshold - scusum_threshold_std / 10),
                                  (scusum_threshold + scusum_threshold_std / 10),
                                  color=color_dict['threshold'], alpha=.1)
                ax_1.vlines(500, ylim_1[0], ylim_1[1], label=label_dict['cp'], color=color_dict['cp'],
                            linestyle=linestyle_dict['cp'])
                ax_1.legend(loc=loc_dict['score'], fontsize=fontsize_dict['legend'])

                x = range(1, len(scanb) + 1)
                ax_2 = ax_dict_2[fig_name]
                xlabel = 'Time Step'
                ylabel = 'Detection Score'
                pivot = 'scanb'
                ax_2.plot(x, scanb, label=label_dict[pivot], color=color_dict[pivot],
                          linestyle=linestyle_dict[pivot])
                ax_2.fill_between(x, (scanb - scanb_std / 10), (scanb + scanb_std / 10), color=color_dict[pivot],
                                  alpha=.1)
                ax_2.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
                ax_2.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
                ax_2.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
                ax_2.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
                ylim_2 = list(ax_2.get_ylim())
                ylim_2[1] = max(ylim_2[1], max(scanb_threshold))
                ax_2.plot(x, scanb_threshold, label=label_dict['threshold'], color=color_dict['threshold'],
                          linestyle=linestyle_dict['threshold'])
                ax_2.fill_between(x, (scanb_threshold - scanb_threshold_std / 10),
                                  (scanb_threshold + scanb_threshold_std / 10), color=color_dict['threshold'], alpha=.1)
                ax_2.vlines(500, ylim_2[0], ylim_2[1], label=label_dict['cp'], color=color_dict['cp'],
                            linestyle=linestyle_dict['cp'])
                ax_2.legend(loc=loc_dict['score'], fontsize=fontsize_dict['legend'])

                x = range(1, len(calm) + 1)
                ax_3 = ax_dict_3[fig_name]
                xlabel = 'Time Step'
                ylabel = 'Detection Score'
                pivot = 'calm'
                ax_3.plot(x, calm, label=label_dict[pivot], color=color_dict[pivot],
                          linestyle=linestyle_dict[pivot])
                ax_3.fill_between(x, (calm - calm_std / 10), (calm + calm_std / 10), color=color_dict[pivot],
                                  alpha=.1)
                ax_3.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
                ax_3.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
                ax_3.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
                ax_3.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
                ylim_3 = list(ax_3.get_ylim())
                ylim_3[1] = max(ylim_3[1], max(calm_threshold))
                ax_3.plot(x, calm_threshold, label=label_dict['threshold'], color=color_dict['threshold'],
                          linestyle=linestyle_dict['threshold'])
                ax_3.fill_between(x, (calm_threshold - calm_threshold_std / 10),
                                  (calm_threshold + calm_threshold_std / 10), color=color_dict['threshold'], alpha=.1)
                ax_3.vlines(500, ylim_3[0], ylim_3[1], label=label_dict['cp'], color=color_dict['cp'],
                            linestyle=linestyle_dict['cp'])
                ax_3.legend(loc=loc_dict['score'], fontsize=fontsize_dict['legend'])
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_2[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_3[fig_name].grid(linestyle='--', linewidth='0.5')
        if fig_name in ax_dict_4:
            ax_dict_4[fig_name].grid(linestyle='--', linewidth='0.5')
        fig[fig_name].tight_layout()
        dir_name = 'score'
        dir_path = os.path.join(vis_path, dir_name)
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_vis_change(df_mean):
    label_dict = {'cusum': 'CUSUM', 'scusum': 'SCUSUM', 'scanb': 'Scan B-statistic', 'calm': 'CALM-MMD'}
    color_dict = {'cusum': 'black', 'scusum': 'red', 'scanb': 'blue', 'calm': 'cyan'}
    linestyle_dict = {'cusum': '-', 'scusum': '--', 'scanb': ':', 'calm': '-.'}
    marker_dict = {'cusum': 'D', 'scusum': 'o', 'scanb': 'p', 'calm': 's'}
    fontsize_dict = {'legend': 16, 'label': 16, 'ticks': 16}
    loc_dict = {'change': 'upper right'}
    fig_names = tree()
    for df_name in df_mean:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        data_name = df_name_list[0]
        change = df_name_list[3]
        noise = df_name_list[4]
        test_mode = df_name_list[5]
        arl = df_name_list[6]
        mask_mvn_mean = 'MVN' in data_name and change.split('-')[1] == '0.0' and \
                        noise == '0' and arl == '2000' and metric_name == 'CADD' and stat == 'mean'
        mask_mvn_logvar = 'MVN' in data_name and change.split('-')[0] == '0.0' and \
                          noise == '0' and arl == '2000' and metric_name == 'CADD' and stat == 'mean'
        mask_exp_tau = 'EXP' in data_name and noise == '0' and arl == '2000' and metric_name == 'CADD' and stat == 'mean'
        mask_rbm_tau = 'RBM' in data_name and noise == '0' and arl == '2000' and metric_name == 'CADD' and stat == 'mean'
        if mask_mvn_mean:
            if change == '0.0-0.0':
                fig_names_ = ['mvn-mean', 'mvn-logvar']
            else:
                fig_names_ = ['mvn-mean']
            x = float(change.split('-')[0])
        elif mask_mvn_logvar:
            fig_names_ = ['mvn-logvar']
            x = float(change.split('-')[1])
        elif mask_exp_tau:
            fig_names_ = ['exp-tau']
            x = float(change.split('-')[0])
        elif mask_rbm_tau:
            fig_names_ = ['rbm-W']
            x = float(change.split('-')[0])
        else:
            continue
        df_name_std = '_'.join([*df_name_list[:-1], 'std'])
        y = df_mean[df_name].iloc[0].to_numpy().item()
        y_std = df_mean[df_name_std].iloc[0].to_numpy().item()
        for fig_name in fig_names_:
            if test_mode not in fig_names[fig_name]:
                fig_names[fig_name][test_mode] = defaultdict(list)
            fig_names[fig_name][test_mode]['x'].append(x)
            fig_names[fig_name][test_mode]['y'].append(y)
            fig_names[fig_name][test_mode]['y_std'].append(y_std)
    fig = {}
    ax_dict_1 = {}
    figsize = (5, 4)
    xlabel_dict = {'mvn-mean': '$\Delta \mu$', 'mvn-logvar': '$\Delta \log(\sigma^2)$', 'exp-tau': '$\Delta \\tau$',
                   'rbm-W': '$\sigma_{\Delta}$'}
    for fig_name in fig_names:
        for test_mode in fig_names[fig_name]:
            x, y, y_std = fig_names[fig_name][test_mode]['x'], fig_names[fig_name][test_mode]['y'], \
                          fig_names[fig_name][test_mode]['y_std']
            x, y, y_std = zip(*sorted(zip(x, y, y_std)))
            x, y, y_std = np.array(x), np.array(y), np.array(y_std)
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
            ax_1 = ax_dict_1[fig_name]
            xlabel = xlabel_dict[fig_name]
            ylabel = 'Empirical CADD'
            pivot = test_mode
            ax_1.errorbar(x, y, yerr=y_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot])
            ax_1.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_1.legend(loc=loc_dict['change'], fontsize=fontsize_dict['legend'])
            ax_1.set_yscale('log')
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        fig[fig_name].tight_layout()
        dir_name = 'change'
        dir_path = os.path.join(vis_path, dir_name)
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_vis_arl(df_mean):
    label_dict = {'cusum': 'CUSUM', 'scusum': 'SCUSUM', 'scanb': 'Scan B-statistic', 'calm': 'CALM-MMD'}
    color_dict = {'cusum': 'black', 'scusum': 'red', 'scanb': 'blue', 'calm': 'cyan'}
    linestyle_dict = {'cusum': '-', 'scusum': '--', 'scanb': ':', 'calm': '-.'}
    marker_dict = {'cusum': 'D', 'scusum': 'o', 'scanb': 'p', 'calm': 's'}
    fontsize_dict = {'legend': 16, 'label': 16, 'ticks': 16}
    loc_dict = {'arl': 'upper right'}
    fig_names = tree()
    for df_name in df_mean:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        data_name = df_name_list[0]
        change = df_name_list[3]
        noise = df_name_list[4]
        test_mode = df_name_list[5]
        arl = df_name_list[6]
        mask_mvn_mean_0 = 'MVN' in data_name and change == '0.0-0.0' and \
                          noise == '0' and metric_name == 'CADD' and stat == 'mean'
        mask_mvn_mean_1 = 'MVN' in data_name and change == '0.1-0.0' and \
                          noise == '0' and metric_name == 'CADD' and stat == 'mean'
        mask_mvn_logvar_1 = 'MVN' in data_name and change == '0.0-0.5' and \
                            noise == '0' and metric_name == 'CADD' and stat == 'mean'
        mask_exp_tau_0 = 'EXP' in data_name and change == '0.0' and \
                         noise == '0' and metric_name == 'CADD' and stat == 'mean'
        mask_exp_tau_1 = 'EXP' in data_name and change == '1.0' and \
                         noise == '0' and metric_name == 'CADD' and stat == 'mean'
        mask_rbm_tau_0 = 'RBM' in data_name and change == '0.0' and \
                         noise == '0' and metric_name == 'CADD' and stat == 'mean'
        mask_rbm_tau_1 = 'RBM' in data_name and change == '0.05' and \
                         noise == '0' and metric_name == 'CADD' and stat == 'mean'
        if mask_mvn_mean_0:
            fig_names_ = ['mvn-mean', 'mvn-logvar']
            mode = '0'
        elif mask_mvn_mean_1:
            fig_names_ = ['mvn-mean']
            mode = '1'
        elif mask_mvn_logvar_1:
            fig_names_ = ['mvn-logvar']
            mode = '1'
        elif mask_exp_tau_0:
            fig_names_ = ['exp-tau']
            mode = '0'
        elif mask_exp_tau_1:
            fig_names_ = ['exp-tau']
            mode = '1'
        elif mask_rbm_tau_0:
            fig_names_ = ['rbm-W']
            mode = '0'
        elif mask_rbm_tau_1:
            fig_names_ = ['rbm-W']
            mode = '1'
        else:
            continue
        df_name_std = '_'.join([*df_name_list[:-1], 'std'])
        x = float(arl)
        y = df_mean[df_name].iloc[0].to_numpy().item()
        y_std = df_mean[df_name_std].iloc[0].to_numpy().item()
        for fig_name in fig_names_:
            if test_mode not in fig_names[fig_name]:
                fig_names[fig_name][test_mode]['0'] = defaultdict(list)
                fig_names[fig_name][test_mode]['1'] = defaultdict(list)
            fig_names[fig_name][test_mode][mode]['x'].append(x)
            fig_names[fig_name][test_mode][mode]['y'].append(y)
            fig_names[fig_name][test_mode][mode]['y_std'].append(y_std)
    fig = {}
    ax_dict_1 = {}
    ax_dict_2 = {}
    figsize = (10, 4)
    for fig_name in fig_names:
        for test_mode in fig_names[fig_name]:
            x_1, y_1, y_1_std = fig_names[fig_name][test_mode]['1']['x'], fig_names[fig_name][test_mode]['1']['y'], \
                                fig_names[fig_name][test_mode]['1']['y_std']
            x_0, y_0, y_0_std = fig_names[fig_name][test_mode]['0']['x'], fig_names[fig_name][test_mode]['0']['y'], \
                                fig_names[fig_name][test_mode]['0']['y_std']
            x_1, y_1, y_1_std = zip(*sorted(zip(x_1, y_1, y_1_std)))
            x_1, y_1, y_1_std = np.array(x_1), np.array(y_1), np.array(y_1_std)
            x_0, y_0, y_0_std = zip(*sorted(zip(x_0, y_0, y_0_std)))
            x_0, y_0, y_0_std = np.array(x_0), np.array(y_0), np.array(y_0_std)
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(121)
                ax_dict_2[fig_name] = fig[fig_name].add_subplot(122)
            ax_1 = ax_dict_1[fig_name]
            ax_2 = ax_dict_2[fig_name]
            xlabel = 'ARL'
            ylabel = 'Empirical CADD'
            pivot = test_mode
            ax_1.errorbar(x_1, y_1, yerr=y_1_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot])
            ax_1.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_1.legend(loc=loc_dict['arl'], fontsize=fontsize_dict['legend'])
            ax_1.set_xscale('log')
            ax_1.set_yscale('log')

            xlabel = 'ARL'
            ylabel = 'Empirical ARL'
            ax_2.errorbar(x_0, y_0, yerr=y_0_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot])
            ax_2.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
            ax_2.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
            ax_2.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_2.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_2.legend(loc=loc_dict['arl'], fontsize=fontsize_dict['legend'])
            ax_2.set_xscale('log')
            ax_2.set_yscale('log')
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_2[fig_name].grid(linestyle='--', linewidth='0.5')
        fig[fig_name].tight_layout()
        dir_name = 'arl'
        dir_path = os.path.join(vis_path, dir_name)
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_vis_noise(df_mean):
    label_dict = {'cusum': 'CUSUM', 'scusum': 'SCUSUM', 'scanb': 'Scan B-statistic', 'calm': 'CALM-MMD'}
    color_dict = {'cusum': 'black', 'scusum': 'red', 'scanb': 'blue', 'calm': 'cyan'}
    linestyle_dict = {'cusum': '-', 'scusum': '--', 'scanb': ':', 'calm': '-.'}
    marker_dict = {'cusum': 'D', 'scusum': 'o', 'scanb': 'p', 'calm': 's'}
    fontsize_dict = {'legend': 16, 'label': 16, 'ticks': 16}
    loc_dict = {'noise': 'upper right'}
    fig_names = tree()
    for df_name in df_mean:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        data_name = df_name_list[0]
        change = df_name_list[3]
        noise = df_name_list[4]
        test_mode = df_name_list[5]
        arl = df_name_list[6]
        mask_mvn_mean_0 = 'MVN' in data_name and change == '0.0-0.0' and \
                          arl == '2000' and metric_name == 'CADD' and stat == 'mean'
        mask_mvn_mean_1 = 'MVN' in data_name and change == '0.1-0.0' and \
                          arl == '2000' and metric_name == 'CADD' and stat == 'mean'
        mask_mvn_logvar_1 = 'MVN' in data_name and change == '0.0-0.5' and \
                            arl == '2000' and metric_name == 'CADD' and stat == 'mean'
        mask_exp_tau_0 = 'EXP' in data_name and change == '0.0' and \
                         arl == '2000' and metric_name == 'CADD' and stat == 'mean'
        mask_exp_tau_1 = 'EXP' in data_name and change == '1.0' and \
                         arl == '2000' and metric_name == 'CADD' and stat == 'mean'
        mask_rbm_tau_0 = 'RBM' in data_name and change == '0.0' and \
                         arl == '2000' and metric_name == 'CADD' and stat == 'mean'
        mask_rbm_tau_1 = 'RBM' in data_name and change == '0.05' and \
                         arl == '2000' and metric_name == 'CADD' and stat == 'mean'
        if mask_mvn_mean_0:
            fig_names_ = ['mvn-mean', 'mvn-logvar']
            mode = '0'
        elif mask_mvn_mean_1:
            fig_names_ = ['mvn-mean']
            mode = '1'
        elif mask_mvn_logvar_1:
            fig_names_ = ['mvn-logvar']
            mode = '1'
        elif mask_exp_tau_0:
            fig_names_ = ['exp-tau']
            mode = '0'
        elif mask_exp_tau_1:
            fig_names_ = ['exp-tau']
            mode = '1'
        elif mask_rbm_tau_0:
            fig_names_ = ['rbm-W']
            mode = '0'
        elif mask_rbm_tau_1:
            fig_names_ = ['rbm-W']
            mode = '1'
        else:
            continue
        df_name_std = '_'.join([*df_name_list[:-1], 'std'])
        x = float(noise)
        y = df_mean[df_name].iloc[0].to_numpy().item()
        y_std = df_mean[df_name_std].iloc[0].to_numpy().item()
        for fig_name in fig_names_:
            if test_mode not in fig_names[fig_name]:
                fig_names[fig_name][test_mode]['0'] = defaultdict(list)
                fig_names[fig_name][test_mode]['1'] = defaultdict(list)
            fig_names[fig_name][test_mode][mode]['x'].append(x)
            fig_names[fig_name][test_mode][mode]['y'].append(y)
            fig_names[fig_name][test_mode][mode]['y_std'].append(y_std)
    fig = {}
    ax_dict_1 = {}
    ax_dict_2 = {}
    figsize = (10, 4)
    for fig_name in fig_names:
        for test_mode in fig_names[fig_name]:
            x_1, y_1, y_1_std = fig_names[fig_name][test_mode]['1']['x'], fig_names[fig_name][test_mode]['1']['y'], \
                                fig_names[fig_name][test_mode]['1']['y_std']
            x_0, y_0, y_0_std = fig_names[fig_name][test_mode]['0']['x'], fig_names[fig_name][test_mode]['0']['y'], \
                                fig_names[fig_name][test_mode]['0']['y_std']
            if len(y_1) == 0 or len(y_0) == 0:
                continue
            x_1, y_1, y_1_std = zip(*sorted(zip(x_1, y_1, y_1_std)))
            x_1, y_1, y_1_std = np.array(x_1), np.array(y_1), np.array(y_1_std)
            x_0, y_0, y_0_std = zip(*sorted(zip(x_0, y_0, y_0_std)))
            x_0, y_0, y_0_std = np.array(x_0), np.array(y_0), np.array(y_0_std)
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(121)
                ax_dict_2[fig_name] = fig[fig_name].add_subplot(122)
            ax_1 = ax_dict_1[fig_name]
            ax_2 = ax_dict_2[fig_name]
            xlabel = '$\sigma_{noise}$'
            ylabel = 'Empirical CADD'
            pivot = test_mode
            ax_1.errorbar(x_1, y_1, yerr=y_1_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot])
            ax_1.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_1.legend(loc=loc_dict['noise'], fontsize=fontsize_dict['legend'])
            ax_1.set_xscale('log')
            ax_1.set_yscale('log')

            xlabel = '$\sigma_{noise}$'
            ylabel = 'Empirical ARL'
            ax_2.errorbar(x_0, y_0, yerr=y_0_std, color=color_dict[pivot], linestyle=linestyle_dict[pivot],
                          label=label_dict[pivot], marker=marker_dict[pivot])
            ax_2.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
            ax_2.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
            ax_2.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_2.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_2.legend(loc=loc_dict['noise'], fontsize=fontsize_dict['legend'])
            ax_2.set_xscale('log')
            ax_2.set_yscale('log')
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_2[fig_name].grid(linestyle='--', linewidth='0.5')
        fig[fig_name].tight_layout()
        dir_name = 'noise'
        dir_path = os.path.join(vis_path, dir_name)
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()
