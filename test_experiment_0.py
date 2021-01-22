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
parser.add_argument('--cmd_py', default='python3 main.py', type=str)
parser.add_argument('--output_dir', default='/groups/astuart/mlevine/ode_model_error/experiments/debugging9/', type=str)
parser.add_argument('--cmd_job', default='bash', type=str)
parser.add_argument('--conda_env', default='', type=str)
FLAGS = parser.parse_args()


def main(cmd_py, output_dir, cmd_job, datagen, conda_env, **kwargs):

    if conda_env=='':
        conda_env = None

    all_job_fnames = []

    os.makedirs(output_dir, exist_ok=True)

    master_job_file = os.path.join(output_dir,'master_job_file.txt')

    ### Define true system (e.g. L63)
    data_pathname = os.path.join(output_dir, 'l63data.pickle')

    datagen_settings = {'rng_seed': 63,
                        't_transient': 30,
                        't_train': 1005,
                        't_invariant_measure': 100,
                        't_test': 20,
                        't_validate': 20,
                        'n_test_traj': 10,
                        'n_train_traj': 2,
                        'n_validate_traj': 7,
                        'delta_t': 0.001,
                        'data_pathname': data_pathname
                    }

    if datagen:
        generate_data(ode=L63(), **datagen_settings)

    shared_settings = {'data_pathname': data_pathname,
                        'f0_name': 'L63',
                        'input_dim': 3,
                        't_test': 20}

    ## HYBRID PHYSICS RUNS
    combined_settings = { 'modelType': ['discrete', 'continuousInterp'],
                 'rfDim': [200],
                 'tTrain': [100],
                 'usef0': [1],
                 'doResidual': [1],
                 'stateType': ['state', 'stateAndPred'],
                 'dt': [0.001, 0.01, 0.1],
                 'f0eps': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2],
                 'trainNumber': [i for i in range(datagen_settings['n_train_traj'])]
                }
    all_job_fnames += queue_joblist(combined_settings=combined_settings, shared_settings=shared_settings, output_dir=output_dir, master_job_file=master_job_file, cmd=cmd_py, conda_env=conda_env)

    combined_settings['doResidual'] = [0]
    combined_settings['stateType'] = ['stateAndPred']
    all_job_fnames += queue_joblist(combined_settings=combined_settings, shared_settings=shared_settings, output_dir=output_dir, master_job_file=master_job_file, cmd=cmd_py, conda_env=conda_env)

    ## f0-ONLY RUN
    combined_settings['doResidual'] = [1]
    combined_settings['stateType'] = ['state']
    combined_settings['modelType'] = ['f0only']
    all_job_fnames += queue_joblist(combined_settings=combined_settings, shared_settings=shared_settings, output_dir=output_dir, master_job_file=master_job_file, cmd=cmd_py, conda_env=conda_env)

    ## DATA-ONLY RUN
    combined_settings['doResidual'] = [0]
    combined_settings['usef0'] = [0]
    combined_settings['f0eps'] = ['NA']
    combined_settings['stateType'] = ['state']
    combined_settings['modelType'] = ['discrete', 'continuousInterp']
    all_job_fnames += queue_joblist(combined_settings=combined_settings, shared_settings=shared_settings, output_dir=output_dir, master_job_file=master_job_file, cmd=cmd_py, conda_env=conda_env)

    # collect job dirs and enumerate their properties
    summary_df = init_summary_df(combined_settings, all_job_fnames)
    summary_df_name = os.path.join(output_dir, 'summary_df.pickle')
    with open(summary_df_name, "wb") as file:
        pickle.dump(summary_df, file, pickle.HIGHEST_PROTOCOL)

    prioritized_job_sender(all_job_fnames, bash_command=cmd_job)

    if cmd_job=='bash':
        run_summary(output_dir)


def prioritized_job_sender(all_job_fnames, bash_command):
    # start with f0only
    for job_fname in all_job_fnames:
        if 'f0only' in job_fname:
            all_job_fnames.remove(job_fname)
            submit_job(job_fname, bash_command=bash_command)

    # next do data-driven only
    for job_fname in all_job_fnames:
        if 'f0eps-NA' in job_fname:
            all_job_fnames.remove(job_fname)
            submit_job(job_fname, bash_command=bash_command)

    # next do favored experiments
    str_list = ['tTrain-100_', 'rfDim-200_', 'stateType-state_', 'trainNumber-0_']
    for job_fname in all_job_fnames:
        if all(elem in job_fname for elem in str_list):
            all_job_fnames.remove(job_fname)
            submit_job(job_fname, bash_command=bash_command)

    # now send remaining jobs
    for job_fname in all_job_fnames:
        all_job_fnames.remove(job_fname)
        submit_job(job_fname, bash_command=bash_command)

    return

def init_summary_df(combined_settings, all_job_fnames):
    summary_df = pd.DataFrame()
    my_vars = list(combined_settings.keys())
    for jobfile_path in all_job_fnames:
        job_dir = os.path.dirname(jobfile_path)
        # determine the value of each my_var
        var_dict = parse_output_path(job_dir, nm_list=my_vars)
        var_dict['eval_pickle_fname'] = os.path.join(job_dir, 'test_eval.pickle')
        if var_dict['modelType']=='f0only':
            var_dict['type'] = 'f0only'
        else:
            var_dict['type'] = '{}, {}, resid={}'.format(var_dict['modelType'] , var_dict['stateType'], var_dict['doResidual'])
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
    metric_list = ['rmse_total', 't_valid_050', 't_valid_005']
    for dt in summary_df.dt.unique():
        for t in summary_df.tTrain.unique():
            for rfd in summary_df.rfDim.unique():
                plot_output_dir = os.path.join(output_dir, 'summary_plots_dt{dt}_tTrain{t}_rfdim{rfd}'.format(dt=dt, t=t, rfd=rfd))
                os.makedirs(plot_output_dir, exist_ok=True)
                try:
                    summarize_eps(df=summary_df[(summary_df.dt==dt) & (summary_df.tTrain==t) & (summary_df.rfDim==rfd)], style='usef0', hue='type', output_dir=plot_output_dir, metric_list=metric_list)
                except:
                    print('plot failed for:', plot_output_dir)


if __name__ == '__main__':
    if FLAGS.mode=='all':
        main(**FLAGS.__dict__)
    elif FLAGS.mode=='plot':
        run_summary(output_dir=FLAGS.output_dir)

# def build_job(settings, settings_path, submit_job, fake=True):
#     command_flag_dict = {'settings_path': settings_path}
#     jobfile_dir = settings_path.strip('.json')
#     jobstatus, jobnum = make_and_deploy(bash_run_command=CMD_run_fits,
#         command_flag_dict=command_flag_dict,
#         jobfile_dir=jobfile_dir,
#         master_job_file=master_job_file, no_submit=submit_job)
#
#     if jobstatus!=0:
#         print('Quitting because job failed!')
#     else:
#         return jobfile_dir




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

## training set size vs {t_valid, KL, ACF-error} w/ errorbars for a given DT and f0-eps
