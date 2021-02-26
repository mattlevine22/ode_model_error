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
parser.add_argument('--output_dir', default='/groups/astuart/mlevine/ode_model_error/experiments/l63eps_v5_rfDim_validateReg/', type=str)
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
                        'n_train_traj': 5,
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
        lop = [['trainNumber-0']
               ]
        prioritized_job_sender(all_job_fnames,
                                bash_command=cmd_job,
                                list_of_priorities=lop,
                                do_all=True,
                                noredo=False)

    run_summary(output_dir)

def declare_jobs(data_pathname, datagen_settings, output_dir, master_job_file, cmd_py, conda_env, hours):
    all_job_fnames = []
    shared_settings = {'data_pathname': data_pathname,
                        'f0_name': 'L63',
                        'input_dim': 3,
                        't_test': 20,
                        't_inv': 100}

    f0eps_list = [3.0]
    dt_list = [0.001]
    rfDim_list = [5, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    ## rhs runs
    combined_settings = { 'modelType': ['rhs'],
                 'diff': ['Spline'],
                 'costIntegration': ['datagrid'],
                 'rfDim': rfDim_list,
                 'tTrain': [100],
                 'usef0': [0],
                 'doResidual': [0],
                 'stateType': ['state'],
                 'dt': dt_list,
                 'f0eps': [0],
                 'trainNumber': [i for i in range(datagen_settings['n_train_traj'])]
                }
    all_job_fnames += queue_joblist(combined_settings=combined_settings, shared_settings=shared_settings, output_dir=output_dir, master_job_file=master_job_file, cmd=cmd_py, conda_env=conda_env, hours=hours)

    # hybrid rhs runs
    combined_settings['usef0'] = [1]
    combined_settings['f0eps'] = f0eps_list

    combined_settings['doResidual'] = [1]
    combined_settings['stateType'] = ['state']
    all_job_fnames += queue_joblist(combined_settings=combined_settings, shared_settings=shared_settings, output_dir=output_dir, master_job_file=master_job_file, cmd=cmd_py, conda_env=conda_env, hours=hours)

    combined_settings['doResidual'] = [0]
    combined_settings['stateType'] = ['stateAndPred']
    all_job_fnames += queue_joblist(combined_settings=combined_settings, shared_settings=shared_settings, output_dir=output_dir, master_job_file=master_job_file, cmd=cmd_py, conda_env=conda_env, hours=hours)

    ## discrete runs
    combined_settings['modelType'] = ['Psi']
    combined_settings['diff'] = ['NA']
    combined_settings['costIntegration'] = ['Psi']

    # hybrid discrete runs
    combined_settings['usef0'] = [1]
    combined_settings['f0eps'] = f0eps_list

    combined_settings['doResidual'] = [1]
    combined_settings['stateType'] = ['state']
    all_job_fnames += queue_joblist(combined_settings=combined_settings, shared_settings=shared_settings, output_dir=output_dir, master_job_file=master_job_file, cmd=cmd_py, conda_env=conda_env, hours=hours)

    combined_settings['doResidual'] = [0]
    combined_settings['stateType'] = ['stateAndPred']
    all_job_fnames += queue_joblist(combined_settings=combined_settings, shared_settings=shared_settings, output_dir=output_dir, master_job_file=master_job_file, cmd=cmd_py, conda_env=conda_env, hours=hours)


    # data-only discrete runs
    combined_settings['usef0'] = [0]
    combined_settings['f0eps'] = [0]
    combined_settings['doResidual'] = [0]
    combined_settings['stateType'] = ['state']
    all_job_fnames += queue_joblist(combined_settings=combined_settings, shared_settings=shared_settings, output_dir=output_dir, master_job_file=master_job_file, cmd=cmd_py, conda_env=conda_env, hours=hours)

    # true-model run
    combined_settings['usef0'] = [1]
    combined_settings['f0eps'] = [0]
    combined_settings['modelType'] = ['f0only']
    combined_settings['diff'] = ['TrueDeriv']
    combined_settings['costIntegration'] = ['fTrue']
    all_job_fnames += queue_joblist(combined_settings=combined_settings, shared_settings=shared_settings, output_dir=output_dir, master_job_file=master_job_file, cmd=cmd_py, conda_env=conda_env, hours=hours)

    # bad-model runs
    combined_settings['usef0'] = [1]
    combined_settings['f0eps'] = f0eps_list
    combined_settings['modelType'] = ['f0only']
    combined_settings['diff'] = ['NA']
    combined_settings['costIntegration'] = ['f0only']
    all_job_fnames += queue_joblist(combined_settings=combined_settings, shared_settings=shared_settings, output_dir=output_dir, master_job_file=master_job_file, cmd=cmd_py, conda_env=conda_env, hours=hours)


    return all_job_fnames, combined_settings

def init_summary_df(combined_settings, all_job_fnames):
    fidelity_list = ['Euler', 'default', 'lowfi', 'medfi', 'hifi', 'hifiPlus']
    summary_df = pd.DataFrame()
    my_vars = list(combined_settings.keys())
    for jobfile_path in all_job_fnames:
        job_dir = os.path.dirname(jobfile_path)
        # determine the value of each my_var
        var_dict = parse_output_path(job_dir, nm_list=my_vars)
        var_dict['eval_pickle_fname'] = os.path.join(job_dir, 'test_eval.pickle')
        var_dict['model_fname'] = os.path.join(job_dir, 'Trained_Models/data.pickle')
        if var_dict['modelType']=='f0only':
            var_dict['type'] = 'resid=1, state'
            var_dict['rhsname'] = 'f0only'
        elif var_dict['modelType']=='rhs':
            var_dict['type'] = 'resid={}, {}'.format(var_dict['doResidual'], var_dict['stateType'])
            var_dict['rhsname'] = 'rhs w/ diff={}, costInt={}'.format( var_dict['diff'], var_dict['costIntegration'])
        elif var_dict['modelType']=='Psi':
            var_dict['type'] = 'resid={}, {}'.format(var_dict['doResidual'], var_dict['stateType'])
            var_dict['rhsname'] = 'Psi'

        # default test_eval.pickle is hifi
        var_dict['fidelity'] = 'hifi'
        summary_df = summary_df.append(var_dict, ignore_index=True)

        # now read fidelity specific test outputs
        for fidelity in fidelity_list:
            var_dict['fidelity'] = fidelity
            var_dict['eval_pickle_fname'] = os.path.join(job_dir, 'test_eval_{}.pickle'.format(fidelity))
            var_dict['model_fname'] = os.path.join(job_dir, 'Trained_Models/data.pickle')
            if var_dict['modelType']=='f0only':
                var_dict['type'] = 'resid=1, state'
                var_dict['rhsname'] = 'f0only'
            elif var_dict['modelType']=='rhs':
                var_dict['type'] = 'resid={}, {}'.format(var_dict['doResidual'], var_dict['stateType'])
                var_dict['rhsname'] = 'rhs w/ diff={}, costInt={}'.format( var_dict['diff'], var_dict['costIntegration'])
            elif var_dict['modelType']=='Psi':
                var_dict['type'] = 'resid={}, {}'.format(var_dict['doResidual'], var_dict['stateType'])
                var_dict['rhsname'] = 'Psi'
            summary_df = summary_df.append(var_dict, ignore_index=True)

    # add epsilons
    new_df = pd.DataFrame()
    new_df['f0eps'] = [e for e in summary_df.f0eps.unique() if isinstance(e, float)]
    new_df['usef0'] = 0
    f0_df = pd.merge(summary_df.loc[summary_df.usef0==0, summary_df.columns != 'f0eps'], new_df, on='usef0', how='left')
    summary_df = pd.concat([summary_df.loc[summary_df.usef0==1], f0_df], sort=False)

    # order the categorical data
    summary_df.fidelity = pd.Categorical(summary_df.fidelity, categories=fidelity_list, ordered=True)
    return summary_df

def run_summary(output_dir):
    full_summary_df_name = os.path.join(output_dir, 'full_summary_df.pickle')
    if os.path.exists(full_summary_df_name):
        with open(full_summary_df_name, "rb") as file:
            summary_df = pickle.load(file)
    else:
        summary_df_name = os.path.join(output_dir, 'summary_df.pickle')
        with open(summary_df_name, "rb") as file:
            summary_df = pickle.load(file)
        summary_df = df_eval(df=summary_df)
        with open(full_summary_df_name, "wb") as file:
            pickle.dump(summary_df, file, pickle.HIGHEST_PROTOCOL)

    # subset summary
    # summary_df = summary_df[(summary_df.modelType!='Psi')]
    summary_df.differentiation_error = summary_df.differentiation_error.astype(float)

    ## build a custom, legible rfDim plot
    # TrueDeriv-DataGrid (rhs idealized)
    # spline w/ DataGrid (rhs practical)
    # f0only (physics only)
    # Psi
    rhsname_list = ['rhs w/ diff=TrueDeriv, costInt=datagrid', 'rhs w/ diff=Spline, costInt=datagrid', 'f0only', 'Psi']
    ## Epsilon-based summary
    sub_df1 = summary_df[(summary_df.stateType!='stateAndPred') & (summary_df.eval_pickle_fname.str.contains("test_eval.pickle"))]
    for f0eps in summary_df.f0eps.unique():
        for fid in summary_df.fidelity.unique():
            for dt in summary_df.dt.unique():
                for t in summary_df.tTrain.unique():
                    sub_df = sub_df1[(sub_df1.dt==dt) & (sub_df1.f0eps==f0eps) & (sub_df1.fidelity==fid) & (sub_df1.tTrain==t)]
                    plot_output_dir = os.path.join(output_dir, 'summary_rfDim_plotsALL_dt{dt}_tTrain{t}_fid{fid}_f0eps{f0eps}'.format(dt=dt, t=t, fid=fid, f0eps=f0eps))
                    os.makedirs(plot_output_dir, exist_ok=True)
                    try:
                        summarize(df=sub_df, style='type', hue='rhsname', x="rfDim", output_dir=plot_output_dir, metric_list=metric_list, fname_shape='rfDim_{}')
                        summarize(df=sub_df[sub_df.rhsname.isin(rhsname_list)], style='type', hue='rhsname', x="rfDim", output_dir=plot_output_dir, metric_list=metric_list, fname_shape='rfDim_legible_{}')
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
