import os
import numpy as np
import pandas as pd
import pickle

from plotting_utils import new_box
import pdb

import argparse

# CMD_generate_data_wrapper = 'python3 $HOME/mechRNN/experiments/scripts/generate_data_wrapper.py'
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='/Users/matthewlevine/Downloads/', type=str)
parser.add_argument('--plot_dir', default='/Users/matthewlevine/Dropbox/mechanistic+ML/active_writeups/2021_ICML/l63_v5_figs/auto', type=str)
FLAGS = parser.parse_args()


####### L96MS EXPERIMENT
def plot_l96ScaleSep(base_dir, plot_dir):
    output_dir = os.path.join(base_dir, 'l96ms_ScaleSep_v0')

    full_summary_df_name = os.path.join(output_dir, 'full_summary_df.pickle')
    with open(full_summary_df_name, "rb") as file:
        summary_df = pickle.load(file)


    # add paper names
    summary_df['Model'] = summary_df.longname
    summary_df['Uses $f_0$'] = 'No'
    summary_df.loc[summary_df.usef0==1, 'Uses $f_0$'] = 'Yes'

    # rhsname_list = ['rhs w/ diff=TrueDeriv, costInt=datagrid', 'rhs w/ diff=Spline, costInt=datagrid', 'f0only', 'Psi']
    # papername_list = ['rhs w/ diff=Spline, costInt=datagrid', 'f0only', 'Psi']
    # papername_list = ['rhs w/ diff=Spline, costInt=datagrid CW resid=1, state',
    #                     'rhs w/ diff=Spline, costInt=datagrid resid=1, state',
    #                     'rhs w/ diff=Spline, costInt=datagrid resid=0, state',
    #                     'Psi CW resid=1, state',
    #                     'Psi CW resid=0, state',
    #                     'Psi resid=1, state',
    #                     'f0only f0only']

    eval_dict = {'t_valid_005': 'Validity Time', 'kl_all': 'KL-Divergence', 'acf_error': 'Autocorrelation Error'}

    papername_dict = {'f0only resid=1, state': '$f^\dag \\approx f_0$',
                        'Psi resid=0, state': '$\Psi^\dag \\approx m$',
                        'rhs w/ diff=Spline, costInt=datagrid resid=0, state': '$f^\dag \\approx m$',
                        'Psi CW resid=1, state': '$\Psi^\dag \\approx \Psi_0 + m$',
                        'rhs w/ diff=Spline, costInt=datagrid CW resid=1, state': '$f^\dag \\approx f_0 + m$'
                        }

    for nm in papername_dict:
        summary_df.loc[summary_df.longname==nm, 'Model'] = papername_dict[nm]

    fid='hifi'
    dt=0.001
    tTrain=100
    paper_df = summary_df[(summary_df.longname.isin(papername_dict.keys())) &
                        (summary_df.stateType!='stateAndPred') &
                        (summary_df.fidelity==fid) &
                        (summary_df.dt==dt) &
                        (summary_df.tTrain==tTrain) &
                        (summary_df.eval_pickle_fname.str.contains("test_eval_{}.pickle".format(fid)))]

    for key in eval_dict:
        fig_path = os.path.join(plot_dir, 'l96_ScaleSep_{}'.format(key))
        new_box(df=paper_df, fig_path=fig_path,
                    x="Model",
                    y=key,
                    order=[papername_dict[k] for k in papername_dict.keys()],
                    xlabel='Models',
                    ylabel=eval_dict[key],
                    title='Model performance comparisons ({})'.format(eval_dict[key]),
                    legloc='best',
                    figsize=(12,12))

if __name__ == '__main__':
    os.makedirs(FLAGS.plot_dir, exist_ok=True)
    plot_l96ScaleSep(**FLAGS.__dict__)
