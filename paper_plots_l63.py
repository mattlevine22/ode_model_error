import os
import numpy as np
import pandas as pd
import pickle

from plotting_utils import new_summary
import pdb

import argparse

# CMD_generate_data_wrapper = 'python3 $HOME/mechRNN/experiments/scripts/generate_data_wrapper.py'
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='/Users/matthewlevine/Downloads/', type=str)
parser.add_argument('--plot_dir', default='/Users/matthewlevine/Dropbox/mechanistic+ML/active_writeups/2021_ICML/l63_v5_figs/auto', type=str)
FLAGS = parser.parse_args()

EVAL_DICT_ALL = {'t_valid_005': 'Validity Time', 'kl_mean': 'KL-Divergence', 'acf_error': 'Autocorrelation Error'}
EVAL_DICT_SHORT = {'t_valid_005': 'Validity Time'}



def plot_epsilonGP(base_dir, plot_dir, eval_dict=EVAL_DICT_ALL):
    output_dir = os.path.join(base_dir, 'l63eps_v5_epsGP_validateReg')

    full_summary_df_name = os.path.join(output_dir, 'full_summary_df.pickle')
    with open(full_summary_df_name, "rb") as file:
        summary_df = pickle.load(file)


    # add paper names
    summary_df['Model'] = summary_df.modelType
    summary_df['Uses $f_0$'] = 'No'
    summary_df.loc[summary_df.usef0==1, 'Uses $f_0$'] = 'Yes'

    # rhsname_list = ['rhs w/ diff=TrueDeriv, costInt=datagrid', 'rhs w/ diff=Spline, costInt=datagrid', 'f0only', 'Psi']
    papername_list = ['rhs w/ diff=Spline, costInt=datagrid', 'f0only', 'Psi']

    fid='hifi'
    dt=0.001
    # rfDim=200
    tTrain=100
    paper_df = summary_df[(summary_df.rhsname.isin(papername_list)) &
                        (summary_df.stateType!='stateAndPred') &
                        (summary_df.fidelity==fid) &
                        (summary_df.dt==dt) &
                        # (summary_df.rfDim==rfDim) &
                        (summary_df.tTrain==tTrain) &
                        (summary_df.eval_pickle_fname.str.contains("test_eval.pickle"))]


    for key in eval_dict:
        fig_path = os.path.join(plot_dir, 'epsGP_quality_{}'.format(key))
        new_summary(df=paper_df, fig_path=fig_path, x="f0eps", y=key,
                    xlabel='$\epsilon$',
                    estimator=np.mean,
                    ci='sd',
                    ylabel=eval_dict[key],
                    title='Effects of model error on hybrid learning',
                    legloc='upper right',
                    figsize=(12,10))


    paper_df = summary_df[(summary_df.rhsname.isin(papername_list)) &
                        (summary_df.stateType=='stateAndPred') &
                        (summary_df.fidelity==fid) &
                        (summary_df.dt==dt) &
                        # (summary_df.rfDim==rfDim) &
                        (summary_df.tTrain==tTrain) &
                        (summary_df.eval_pickle_fname.str.contains("test_eval.pickle"))]


    for key in eval_dict:
        fig_path = os.path.join(plot_dir, 'epsGP_quality_stateAndPred_{}'.format(key))
        new_summary(df=paper_df, fig_path=fig_path, x="f0eps", y=key,
                    xlabel='$\epsilon$',
                    estimator=np.mean,
                    ci='sd',
                    ylabel=eval_dict[key],
                    title='Effects of model error on hybrid learning',
                    legloc='upper right',
                    figsize=(12,10))



####### EPSILON EXPERIMENT
def plot_epsilon(base_dir, plot_dir, eval_dict=EVAL_DICT_ALL):
    output_dir = os.path.join(base_dir, 'l63eps_v5_eps_validateReg')

    full_summary_df_name = os.path.join(output_dir, 'full_summary_df.pickle')
    with open(full_summary_df_name, "rb") as file:
        summary_df = pickle.load(file)


    # add paper names
    summary_df['Model'] = summary_df.modelType
    summary_df['Uses $f_0$'] = 'No'
    summary_df.loc[summary_df.usef0==1, 'Uses $f_0$'] = 'Yes'

    # rhsname_list = ['rhs w/ diff=TrueDeriv, costInt=datagrid', 'rhs w/ diff=Spline, costInt=datagrid', 'f0only', 'Psi']
    papername_list = ['rhs w/ diff=Spline, costInt=datagrid', 'f0only', 'Psi']

    fid='hifi'
    dt=0.001
    # rfDim=200
    tTrain=100
    paper_df = summary_df[(summary_df.rhsname.isin(papername_list)) &
                        (summary_df.stateType!='stateAndPred') &
                        (summary_df.fidelity==fid) &
                        (summary_df.dt==dt) &
                        # (summary_df.rfDim==rfDim) &
                        (summary_df.tTrain==tTrain) &
                        (summary_df.eval_pickle_fname.str.contains("test_eval.pickle"))]


    for key in eval_dict:
        fig_path = os.path.join(plot_dir, 'eps_quality_{}'.format(key))
        new_summary(df=paper_df, fig_path=fig_path, x="f0eps", y=key,
                    xlabel='$\epsilon$',
                    estimator=np.mean,
                    ci='sd',
                    ylabel=eval_dict[key],
                    title='Effects of model error on hybrid learning',
                    legloc='upper right',
                    figsize=(12,10))


    paper_df = summary_df[(summary_df.rhsname.isin(papername_list)) &
                        (summary_df.stateType=='stateAndPred') &
                        (summary_df.fidelity==fid) &
                        (summary_df.dt==dt) &
                        # (summary_df.rfDim==rfDim) &
                        (summary_df.tTrain==tTrain) &
                        (summary_df.eval_pickle_fname.str.contains("test_eval.pickle"))]


    for key in eval_dict:
        fig_path = os.path.join(plot_dir, 'eps_quality_stateAndPred_{}'.format(key))
        new_summary(df=paper_df, fig_path=fig_path, x="f0eps", y=key,
                    xlabel='$\epsilon$',
                    estimator=np.mean,
                    ci='sd',
                    ylabel=eval_dict[key],
                    title='Effects of model error on hybrid learning',
                    legloc='upper right',
                    figsize=(12,10))




####### DT EXPERIMENT
def plot_dt(base_dir, plot_dir, eval_dict=EVAL_DICT_ALL):
    output_dir = os.path.join(base_dir, 'l63eps_v5_dt_validateReg2')

    full_summary_df_name = os.path.join(output_dir, 'full_summary_df.pickle')
    with open(full_summary_df_name, "rb") as file:
        summary_df = pickle.load(file)


    # add paper names
    summary_df['Model'] = summary_df.modelType
    summary_df['Uses $f_0$'] = 'No'
    summary_df.loc[summary_df.usef0==1, 'Uses $f_0$'] = 'Yes'

    # rhsname_list = ['rhs w/ diff=TrueDeriv, costInt=datagrid', 'rhs w/ diff=Spline, costInt=datagrid', 'f0only', 'Psi']
    papername_list = ['rhs w/ diff=Spline, costInt=datagrid', 'f0only', 'Psi']

    fid='hifi'
    f0eps=0.05
    # rfDim=200
    tTrain=1000
    paper_df = summary_df[(summary_df.rhsname.isin(papername_list)) &
                        (summary_df.stateType!='stateAndPred') &
                        (summary_df.fidelity==fid) &
                        (summary_df.f0eps==f0eps) &
                        # (summary_df.rfDim==rfDim) &
                        (summary_df.tTrain==tTrain) &
                        (summary_df.eval_pickle_fname.str.contains("test_eval.pickle"))]

    for key in eval_dict:
        fig_path = os.path.join(plot_dir, 'dt_quality_{}'.format(key))
        new_summary(df=paper_df, fig_path=fig_path, x="dt", y=key,
                    xlabel='$\Delta t$',
                    ylabel=eval_dict[key],
                    estimator=np.mean,
                    ci='sd',
                    title='Effects of sampling rate on learning methods',
                    legloc='upper right',
                    figsize=(12,10))


####### T EXPERIMENT
def plot_T(base_dir, plot_dir, eval_dict=EVAL_DICT_SHORT):
    output_dir = os.path.join(base_dir, 'l63eps_v5_T_validateReg')

    full_summary_df_name = os.path.join(output_dir, 'full_summary_df.pickle')
    with open(full_summary_df_name, "rb") as file:
        summary_df = pickle.load(file)


    # add paper names
    summary_df['Model'] = summary_df.modelType
    summary_df['Uses $f_0$'] = 'No'
    summary_df.loc[summary_df.usef0==1, 'Uses $f_0$'] = 'Yes'

    # rhsname_list = ['rhs w/ diff=TrueDeriv, costInt=datagrid', 'rhs w/ diff=Spline, costInt=datagrid', 'f0only', 'Psi']
    # papername_list = ['rhs w/ diff=Spline, costInt=datagrid', 'f0only', 'Psi']
    papername_list = ['rhs w/ diff=Spline, costInt=datagrid', 'Psi']


    fid='hifi'
    # OLD SETTINGS
    # f0eps=0.2
    # dt=0.01
    # rfDim=2000
    # NEW SETTINGS
    f0eps=0.05
    dt=0.01
    rfDim=10000

    paper_df = summary_df[(summary_df.rhsname.isin(papername_list)) &
                        (summary_df.stateType!='stateAndPred') &
                        (summary_df.fidelity==fid) &
                        (summary_df.f0eps==f0eps) &
                        (summary_df.dt==dt) &
                        (summary_df.rfDim==rfDim) &
                        (summary_df.eval_pickle_fname.str.contains("test_eval.pickle"))]

    for key in eval_dict:
        fig_path = os.path.join(plot_dir, 'T_quality_{}'.format(key))
        new_summary(df=paper_df, fig_path=fig_path, x="tTrain", y=key,
                    xlabel='Length of training trajectory (T)',
                    ylabel=eval_dict[key],
                    estimator=np.mean,
                    ci='sd',
                    title='Data quantity needed by hybrid learning methods',
                    legloc='best',
                    figsize=(12,10))


####### rfDim EXPERIMENT
def plot_rfDim(base_dir, plot_dir, eval_dict=EVAL_DICT_SHORT):
    output_dir = os.path.join(base_dir, 'l63eps_v5_rfDim_validateReg')

    full_summary_df_name = os.path.join(output_dir, 'full_summary_df.pickle')
    with open(full_summary_df_name, "rb") as file:
        summary_df = pickle.load(file)


    # add paper names
    summary_df['Model'] = summary_df.modelType
    summary_df['Uses $f_0$'] = 'No'
    summary_df.loc[summary_df.usef0==1, 'Uses $f_0$'] = 'Yes'

    # rhsname_list = ['rhs w/ diff=TrueDeriv, costInt=datagrid', 'rhs w/ diff=Spline, costInt=datagrid', 'f0only', 'Psi']
    # papername_list = ['rhs w/ diff=Spline, costInt=datagrid', 'f0only', 'Psi']
    papername_list = ['rhs w/ diff=Spline, costInt=datagrid', 'Psi']

    fid='hifi'
    # f0eps=3.0
    # dt=0.001
    # # f0eps=0.2
    # # dt=0.01
    # tTrain=100
    # NEW SETTINGS
    f0eps=0.05
    dt=0.001
    tTrain=100

    paper_df = summary_df[(summary_df.rhsname.isin(papername_list)) &
                        (summary_df.stateType!='stateAndPred') &
                        (summary_df.fidelity==fid) &
                        (summary_df.f0eps==f0eps) &
                        (summary_df.dt==dt) &
                        (summary_df.tTrain==tTrain) &
                        (summary_df.eval_pickle_fname.str.contains("test_eval.pickle"))]

    fig_path = os.path.join(plot_dir, 'rfDim_quality')

    for key in eval_dict:
        fig_path = os.path.join(plot_dir, 'rfDim_quality_{}'.format(key))
        new_summary(df=paper_df, fig_path=fig_path, x="rfDim", y=key,
                    xlabel='$D_r$',
                    ylabel=eval_dict[key],
                    estimator=np.mean,
                    ci='sd',
                    title='Model complexity needed for hybrid learning',
                    legloc='best',
                    figsize=(12,10))


####### rfDim EXPERIMENT
def plot_rfDim_chua(base_dir, plot_dir, eval_dict=EVAL_DICT_ALL):
    output_dir = os.path.join(base_dir, 'l63_CHUA_v5_rfDim_validateReg')

    full_summary_df_name = os.path.join(output_dir, 'full_summary_df.pickle')
    with open(full_summary_df_name, "rb") as file:
        summary_df = pickle.load(file)


    # add paper names
    summary_df['Model'] = summary_df.modelType
    summary_df['Uses $f_0$'] = 'No'
    summary_df.loc[summary_df.usef0==1, 'Uses $f_0$'] = 'Yes'

    # rhsname_list = ['rhs w/ diff=TrueDeriv, costInt=datagrid', 'rhs w/ diff=Spline, costInt=datagrid', 'f0only', 'Psi']
    # papername_list = ['rhs w/ diff=Spline, costInt=datagrid', 'f0only', 'Psi']
    papername_list = ['rhs w/ diff=Spline, costInt=datagrid', 'Psi']

    fid='hifi'
    # f0eps=3.0
    # dt=0.001
    # # f0eps=0.2
    # # dt=0.01
    # tTrain=100
    # NEW SETTINGS
    f0eps=0
    dt=0.001
    tTrain=100

    paper_df = summary_df[(summary_df.rhsname.isin(papername_list)) &
                        (summary_df.fidelity==fid) &
                        (summary_df.f0eps==f0eps) &
                        (summary_df.dt==dt) &
                        (summary_df.tTrain==tTrain) &
                        (summary_df.eval_pickle_fname.str.contains("test_eval.pickle"))]

    for nm in papername_list:
        paper_df.loc[(paper_df.rhsname == nm) & (paper_df.stateType=='stateAndPred'), "Model"] += ' Augmented Inputs'

    for key in eval_dict:
        fig_path = os.path.join(plot_dir, 'CHUA_rfDim_quality_{}'.format(key))
        new_summary(df=paper_df, fig_path=fig_path, x="rfDim", y=key,
                    xlabel='$D_r$',
                    ylabel=eval_dict[key],
                    estimator=np.mean,
                    ci='sd',
                    title='Model complexity needed for hybrid learning',
                    legloc='best',
                    figsize=(12,10))

def plot_rfDim_waterwheel(base_dir, plot_dir, eval_dict=EVAL_DICT_ALL):
    output_dir = os.path.join(base_dir, 'l63_WATERWHEEL_v5_rfDim_validateAll')

    full_summary_df_name = os.path.join(output_dir, 'full_summary_df.pickle')
    with open(full_summary_df_name, "rb") as file:
        summary_df = pickle.load(file)


    # add paper names
    summary_df['Model'] = summary_df.modelType
    summary_df['$f_0$'] = 'None'
    summary_df.loc[(summary_df.doResidual==0) & (summary_df.stateType=='stateAndPred'), '$f_0$'] = 'Augmented'
    summary_df.loc[(summary_df.doResidual==1) & (summary_df.stateType=='stateAndPred'), '$f_0$'] = 'Augmented residual'
    summary_df.loc[(summary_df.doResidual==1) & (summary_df.stateType=='state'), '$f_0$'] = 'Residual'

    # rhsname_list = ['rhs w/ diff=TrueDeriv, costInt=datagrid', 'rhs w/ diff=Spline, costInt=datagrid', 'f0only', 'Psi']
    # papername_list = ['rhs w/ diff=Spline, costInt=datagrid', 'f0only', 'Psi']
    papername_list = ['rhs w/ diff=Spline, costInt=datagrid', 'Psi']

    fid='hifi'
    # f0eps=3.0
    # dt=0.001
    # # f0eps=0.2
    # # dt=0.01
    # tTrain=100
    # NEW SETTINGS
    f0eps=0
    dt=0.001
    tTrain=100

    paper_df = summary_df[(summary_df.rhsname.isin(papername_list)) &
                        (summary_df.fidelity==fid) &
                        (summary_df.f0eps==f0eps) &
                        (summary_df.dt==dt) &
                        (summary_df.tTrain==tTrain) &
                        (summary_df.eval_pickle_fname.str.contains("test_eval.pickle"))]

    # for nm in papername_list:
    #     paper_df.loc[(paper_df.rhsname == nm) & (paper_df.stateType=='stateAndPred'), "Model"] += ' Augmented Inputs'

    for key in eval_dict:
        fig_path = os.path.join(plot_dir, 'WATERWHEEL_rfDim_quality_{}'.format(key))
        new_summary(df=paper_df, fig_path=fig_path, x="rfDim", y=key,
                    xlabel='$D_r$',
                    ylabel=eval_dict[key],
                    estimator=np.mean,
                    hue='$f_0$', style='Model',
                    ci='sd',
                    title='Model complexity needed for hybrid learning',
                    legloc='best',
                    figsize=(12,10))



if __name__ == '__main__':
    os.makedirs(FLAGS.plot_dir, exist_ok=True)
    plot_rfDim_waterwheel(**FLAGS.__dict__)
    plot_rfDim_chua(**FLAGS.__dict__)
    plot_epsilonGP(**FLAGS.__dict__)
    plot_epsilon(**FLAGS.__dict__)
    plot_dt(**FLAGS.__dict__)
    plot_T(**FLAGS.__dict__)
    plot_rfDim(**FLAGS.__dict__)
