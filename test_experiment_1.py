import os, sys
import numpy as np
from utils import *
from plotting_utils import *
from odelibrary import L63
import pandas as pd
import pdb
import pickle
import argparse

# CMD_generate_data_wrapper = 'python3 $HOME/mechRNN/experiments/scripts/generate_data_wrapper.py'
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='all', type=str)
parser.add_argument('--datagen', default=1, type=int)
parser.add_argument('--regen', default=0, type=int)
parser.add_argument('--cmd_py', default='python3 main.py', type=str)
parser.add_argument('--output_dir', default='/groups/astuart/mlevine/ode_model_error/experiments/l63eps_v2/', type=str)
parser.add_argument('--cmd_job', default='bash', type=str)
parser.add_argument('--conda_env', default='', type=str)
parser.add_argument('--hours', default=2, type=int)
FLAGS = parser.parse_args()


def main(cmd_py, output_dir, cmd_job, datagen, conda_env, **kwargs):

    if conda_env=='':
        conda_env = None

    os.makedirs(output_dir, exist_ok=True)

    master_job_file = os.path.join(output_dir,'master_job_file.txt')

    ### Define true system (e.g. L63)
    data_pathname = os.path.join(output_dir, 'l63data.pickle')

    datagen_settings = {'rng_seed': 63,
                        't_transient': 30,
                        't_train': 105,
                        't_invariant_measure': 100,
                        't_test': 20,
                        't_validate': 20,
                        'n_test_traj': 10,
                        'n_train_traj': 2,
                        'n_validate_traj': 7,
                        'delta_t': 0.0001,
                        'solver_type': 'hifiPlus',
                        'data_pathname': data_pathname
                    }

    if datagen:
        generate_data(ode=L63(), **datagen_settings)

    ## Get Job List
    if kwargs['regen'] or kwargs['mode']=='all':
        all_job_fnames, combined_settings = declare_jobs(data_pathname, datagen_settings, output_dir, master_job_file, cmd_py, conda_env, hours=kwargs['hours'])

        # collect job dirs and enumerate their properties
        summary_df = init_summary_df(combined_settings, all_job_fnames)
        summary_df_name = os.path.join(output_dir, 'summary_df.pickle')
        with open(summary_df_name, "wb") as file:
            pickle.dump(summary_df, file, pickle.HIGHEST_PROTOCOL)

    if kwargs['mode']=='all':
        lop = [['modelType-discrete', 'diff-NA', 'f0eps-NA', 'ZY-old', 'usef0-0', 'doResidual-0', 'dt-0.001', 'trainNumber-0'],
                ['modelType-continuous', 'f0eps-NA', 'ZY-old', 'usef0-0', 'doResidual-0', 'dt-0.001', 'trainNumber-0'],
                ['modelType-Euler', 'f0eps-NA', 'ZY-old', 'usef0-0', 'doResidual-0', 'dt-0.001', 'trainNumber-0']
               ]
        prioritized_job_sender(all_job_fnames,
                                bash_command=cmd_job,
                                list_of_priorities=lop)

    run_summary(output_dir)

def declare_jobs(data_pathname, datagen_settings, output_dir, master_job_file, cmd_py, conda_env, hours):
    all_job_fnames = []
    shared_settings = {'data_pathname': data_pathname,
                        'f0_name': 'L63',
                        'input_dim': 3,
                        't_test': 20}

    ## HYBRID PHYSICS RUNS in continuous-time
    combined_settings = { 'modelType': ['continuous', 'Euler', 'discrete'],
                 'diff': ['InterpThenDiff', 'DiffThenInterp', 'TrueDeriv', 'NA'],
                 'ZY': ['old'],
                 'rfDim': [200],
                 'tTrain': [100],
                 'usef0': [0, 1],
                 'doResidual': [0, 1],
                 'stateType': ['state'],
                 'dt': [0.0001, 0.001, 0.01, 0.1],
                 'f0eps': [0.001, 0.01, 0.1, 1],
                 'trainNumber': [i for i in range(datagen_settings['n_train_traj'])]
                }
    all_job_fnames += queue_joblist(combined_settings=combined_settings, shared_settings=shared_settings, output_dir=output_dir, master_job_file=master_job_file, cmd=cmd_py, conda_env=conda_env, hours=hours)

    return all_job_fnames, combined_settings

def prioritized_job_sender(all_job_fnames, bash_command, list_of_priorities):

    list_of_priorities.append(['']) # this includes all remaining things at the end
    for check_list in list_of_priorities:
        rmv_nms = []
        for job_fname in all_job_fnames:
            if all(elem in job_fname for elem in check_list):
                rmv_nms.append(job_fname)
                submit_job(job_fname, bash_command=bash_command)
        [all_job_fnames.remove(nm) for nm in rmv_nms]

    if len(all_job_fnames):
        [submit_job(j, bash_command=bash_command) for j in all_job_fnames]

    return

def init_summary_df(combined_settings, all_job_fnames):
    summary_df = pd.DataFrame()
    my_vars = list(combined_settings.keys())
    for jobfile_path in all_job_fnames:
        job_dir = os.path.dirname(jobfile_path)
        # determine the value of each my_var
        var_dict = parse_output_path(job_dir, nm_list=my_vars)
        var_dict['eval_pickle_fname'] = os.path.join(job_dir, 'test_eval.pickle')
        var_dict['model_fname'] = os.path.join(job_dir, 'Trained_Models/data.pickle')
        if var_dict['modelType']=='f0only':
            var_dict['type'] = 'f0only'
        else:
            var_dict['type'] = '{}, {}, resid={}, diff={}'.format(var_dict['modelType'] , var_dict['stateType'], var_dict['doResidual'], var_dict['diff'])

        # default test_eval.pickle is hifi
        var_dict['fidelity'] = 'hifi'
        summary_df = summary_df.append(var_dict, ignore_index=True)

        # now read fidelity specific test outputs
        for fidelity in ['default', 'lowfi', 'medfi', 'hifi', 'hifiPlus']:
            var_dict['fidelity'] = fidelity
            var_dict['eval_pickle_fname'] = os.path.join(job_dir, 'test_eval_{}.pickle'.format(fidelity))
            var_dict['model_fname'] = os.path.join(job_dir, 'Trained_Models/data.pickle')
            if var_dict['modelType']=='f0only':
                var_dict['type'] = 'f0only'
            else:
                var_dict['type'] = '{}, {}, resid={}, diff={}'.format(var_dict['modelType'] , var_dict['stateType'], var_dict['doResidual'], var_dict['diff'])
            summary_df = summary_df.append(var_dict, ignore_index=True)

    # add epsilons
    new_df = pd.DataFrame()
    new_df['f0eps'] = [e for e in summary_df.f0eps.unique() if isinstance(e, float)]
    new_df['usef0'] = 0
    f0_df = pd.merge(summary_df.loc[summary_df.usef0==0, summary_df.columns != 'f0eps'], new_df, on='usef0', how='left')
    summary_df = pd.concat([summary_df.loc[summary_df.usef0==1], f0_df], sort=False)
    return summary_df

def run_summary(output_dir):
    summary_df_name = os.path.join(output_dir, 'summary_df.pickle')
    with open(summary_df_name, "rb") as file:
        summary_df = pickle.load(file)
    summary_df = df_eval(df=summary_df)
    metric_list = ['rmse_total', 't_valid_050', 't_valid_005', 'regularization_RF', 'rf_Win_bound', 'rf_bias_bound']

    ## Solver-based summary
    for f0eps in summary_df.f0eps.unique():
        for ZY in summary_df.ZY.unique():
            for t in summary_df.tTrain.unique():
                for rfd in summary_df.rfDim.unique():
                    for dt in summary_df.dt.unique():
                        plot_output_dir = os.path.join(output_dir, 'summary_plots_f0eps{f0eps}_tTrain{t}_rfdim{rfd}_dt{dt}_ZY{ZY}'.format(f0eps=f0eps, t=t, rfd=rfd, dt=dt, ZY=ZY))
                        os.makedirs(plot_output_dir, exist_ok=True)
                        try:
                            summarize(df=summary_df[(summary_df.stateType!='stateAndPred') & (summary_df.f0eps==f0eps) & (summary_df.tTrain==t) & (summary_df.rfDim==rfd) & (summary_df.ZY==ZY)], style='usef0', hue='type', x="fidelity", output_dir=plot_output_dir, metric_list=metric_list, fname_shape='solvers_{}')
                            summarize(df=summary_df[(summary_df.f0eps==f0eps) & (summary_df.tTrain==t) & (summary_df.rfDim==rfd & (summary_df.ZY==ZY))], style='usef0', hue='type', x="fidelity", output_dir=plot_output_dir, metric_list=metric_list, fname_shape='solvers_all_{}')
                        except:
                            print('plot failed for:', plot_output_dir)


    ## Epsilon-based summary
    for dt in summary_df.dt.unique():
        for t in summary_df.tTrain.unique():
            for rfd in summary_df.rfDim.unique():
                plot_output_dir = os.path.join(output_dir, 'summary_plots_dt{dt}_tTrain{t}_rfdim{rfd}'.format(dt=dt, t=t, rfd=rfd))
                os.makedirs(plot_output_dir, exist_ok=True)
                try:
                    summarize(df=summary_df[(summary_df.stateType!='stateAndPred') & (summary_df.dt==dt) & (summary_df.tTrain==t) & (summary_df.rfDim==rfd)], style='usef0', hue='type', x="f0eps", output_dir=plot_output_dir, metric_list=metric_list, fname_shape='eps_{}')
                    summarize(df=summary_df[(summary_df.dt==dt) & (summary_df.tTrain==t) & (summary_df.rfDim==rfd)], style='usef0', hue='type', x="f0eps", output_dir=plot_output_dir, metric_list=metric_list, fname_shape='eps_all_{}')
                except:
                    print('plot failed for:', plot_output_dir)

    ## DeltaT-based summary
    for f0eps in summary_df.f0eps.unique():
        for t in summary_df.tTrain.unique():
            for rfd in summary_df.rfDim.unique():
                plot_output_dir = os.path.join(output_dir, 'summary_plots_f0eps{f0eps}_tTrain{t}_rfdim{rfd}'.format(f0eps=f0eps, t=t, rfd=rfd))
                os.makedirs(plot_output_dir, exist_ok=True)
                try:
                    summarize(df=summary_df[(summary_df.stateType!='stateAndPred') & (summary_df.f0eps==f0eps) & (summary_df.tTrain==t) & (summary_df.rfDim==rfd)], style='usef0', hue='type', x="dt", output_dir=plot_output_dir, metric_list=metric_list, fname_shape='dt_{}')
                    summarize(df=summary_df[(summary_df.f0eps==f0eps) & (summary_df.tTrain==t) & (summary_df.rfDim==rfd)], style='usef0', hue='type', x="dt", output_dir=plot_output_dir, metric_list=metric_list, fname_shape='dt_all_{}')
                except:
                    print('plot failed for:', plot_output_dir)

if __name__ == '__main__':
    # if FLAGS.mode=='all':
    main(**FLAGS.__dict__)
    # elif FLAGS.mode=='plot':
    #     run_summary(output_dir=FLAGS.output_dir)


### Runs
# -f0-only + {w/ f0, w/out f0} X {m(x), m(f0(x)), m(x,f0(x))}
# -continuous via finite difference, continuous via interp-diff-integrate, discrete

  ## For each training trajectory

    # Train model
    # Output training statistics

    ## For each testing trajectory
      # Evaluate model on testing set
      # Output testing statistics


### Make plots

## f0-eps vs {t_valid, KL, ACF-error} w/ errorbars for a given DT and training set size

## DT vs {t_valid, KL, ACF-error} w/ errorbars for a given f0-eps and training set size

## training set size vs {t_valid, KL, ACF-error} w/ errorbars for a given DT and f0-eps
