import os, sys
import numpy as np
from new_utils import *
from esn_plotting_utils import *
from odelibrary import L63
import pandas as pd
import pdb
import pickle
import argparse

# CMD_generate_data_wrapper = 'python3 $HOME/mechRNN/experiments/scripts/generate_data_wrapper.py'
parser = argparse.ArgumentParser()
parser.add_argument('--cmd_py', default='python3 main.py', type=str)
parser.add_argument('--output_dir', default='experiments/debugging2/', type=str)
parser.add_argument('--cmd_job', default='bash', type=str)
FLAGS = parser.parse_args()


def main(cmd_py, output_dir, cmd_job):

    os.makedirs(output_dir, exist_ok=True)

    master_job_file = os.path.join(output_dir,'master_job_file.txt')

    ### Define true system (e.g. L63)
    data_pathname = os.path.join(output_dir, 'l63data.pickle')

    datagen_settings = {'rng_seed': 63,
            			't_transient': 10,
            			't_train': 105,
            			't_invariant_measure': 100,
            			't_test': 20,
            			'n_test_traj': 10,
            			'n_train_traj': 2,
            			'delta_t': 0.001,
                        'data_pathname': data_pathname
            		}

    generate_data(ode=L63(), **datagen_settings)

    shared_settings = {'data_pathname': data_pathname,
                        'f0_name': 'L63',
                        'input_dim': 3,
                        't_train': 100,
                        't_test': 20}

    ## HYBRID PHYSICS RUN
    # combined_settings = { 'modelType': ['continuousInterp', 'continuousFD', 'discrete'],
     # 'stateType': ['state', 'pred', 'stateAndPred'],
    combined_settings = { 'modelType': ['continuousInterp'],
                 'usef0': [1],
                 'stateType': ['state'],
                 'dt': [0.01],
                 'f0eps': [0.001, 0.01, 0.05, 0.1, 0.2],
                 'trainNumber': [i for i in range(datagen_settings['n_train_traj'])]
                }
    job_fname_list1 = queue_joblist(combined_settings=combined_settings, shared_settings=shared_settings, output_dir=output_dir, master_job_file=master_job_file, cmd=cmd_py)

    ## DATA-ONLY RUN
    combined_settings['usef0'] = [0]
    combined_settings['f0eps'] = ['NA']
    job_fname_list2 = queue_joblist(combined_settings=combined_settings, shared_settings=shared_settings, output_dir=output_dir, master_job_file=master_job_file, cmd=cmd_py)

    all_job_fnames = job_fname_list1 + job_fname_list2

    # collect job dirs and enumerate their properties
    summary_df = init_summary_df(combined_settings, all_job_fnames)
    summary_df_name = os.path.join(output_dir, 'summary_df.pickle')
    with open(summary_df_name, "wb") as file:
        pickle.dump(summary_df, file, pickle.HIGHEST_PROTOCOL)

    for job_fname in all_job_fnames:
        submit_job(job_fname, bash_command=cmd_job)

    if cmd_job=='bash':
        summary_df = df_eval(df=summary_df)
        metric_list = ['rmse_total', 'num_accurate_pred_050', 'num_accurate_pred_005']
        summarize_eps(df=summary_df, style='type', hue='usef0', output_dir=output_dir, metric_list=metric_list)


def init_summary_df(combined_settings, all_job_fnames):
    summary_df = pd.DataFrame()
    my_vars = list(combined_settings.keys())
    for jobfile_path in all_job_fnames:
        job_dir = os.path.dirname(jobfile_path)
        # determine the value of each my_var
        var_dict = parse_output_path(job_dir, nm_list=my_vars)
        var_dict['eval_pickle_fname'] = os.path.join(job_dir, 'test_eval.pickle')
        var_dict['type'] = '{}, {}'.format(var_dict['modelType'] , var_dict['stateType'])
        summary_df = summary_df.append(var_dict, ignore_index=True)
    # add epsilons
    new_df = pd.DataFrame()
    new_df['f0eps'] = [e for e in summary_df.f0eps.unique() if isinstance(e, float)]
    new_df['usef0'] = 0.0
    f0_df = pd.merge(summary_df.loc[summary_df.usef0==0.0, summary_df.columns != 'f0eps'], new_df, on='usef0', how='left')
    summary_df = pd.concat([summary_df.loc[summary_df.usef0==1.0], f0_df])
    return summary_df

if __name__ == '__main__':
    main(**FLAGS.__dict__)

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
