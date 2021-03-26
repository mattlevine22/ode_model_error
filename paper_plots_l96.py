import os
import numpy as np
import pandas as pd
import pickle

from plotting_utils import *

import pdb

import argparse

# CMD_generate_data_wrapper = 'python3 $HOME/mechRNN/experiments/scripts/generate_data_wrapper.py'
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='/Users/matthewlevine/Downloads/', type=str)
parser.add_argument('--plot_dir', default='/Users/matthewlevine/Dropbox/mechanistic+ML/active_writeups/2021_ICML/l63_v5_figs/auto', type=str)
FLAGS = parser.parse_args()

####### L96MS EXPERIMENT
def plot_l96eps_v2(base_dir, plot_dir, experiment_dir='l96ms_ScaleSep_v0', trainNumber=0):
    plot_dir += '_v2'

    os.makedirs(plot_dir, exist_ok=True)

    fid='hifi'
    dt=0.001
    tTrain=100
    # eval_dict = {'t_valid_005': 'Validity Time', 'kl_all': 'KL-Divergence', 'acf_error': 'Autocorrelation Error'}
    good_names = ['f0only resid=1, state', 'Psi CW resid=1, state', 'rhs w/ diff=Spline, costInt=datagrid CW resid=1, state']
    papername_dict = {'f0only resid=1, state': '$f^\dag \\approx f_0$',
                        'Psi resid=0, state': '$\Psi^\dag \\approx m$',
                        'rhs w/ diff=Spline, costInt=datagrid resid=0, state': '$f^\dag \\approx m$',
                        'Psi CW resid=1, state': '$\Psi^\dag \\approx \Psi_0 + m$',
                        'rhs w/ diff=Spline, costInt=datagrid CW resid=1, state': '$f^\dag \\approx f_0 + m$'
                        }
    papername_styledict = {'f0only resid=1, state': {'color': 'gray', 'linestyle': ':'},
                        'Psi resid=0, state': {'color': 'orange', 'linestyle': '--'},
                        'rhs w/ diff=Spline, costInt=datagrid resid=0, state': {'color': 'blue', 'linestyle': '--'},
                        'Psi CW resid=1, state': {'color': 'orange', 'linestyle': '-'},
                        'rhs w/ diff=Spline, costInt=datagrid CW resid=1, state': {'color': 'blue', 'linestyle': '-'}
                        }

    ylabel_dict = {'kl_all': 'Probability', 'acf_error': 'ACF', 't_valid_005': 'Validity Time', 't_valid_050': 'Validity Time'}
	# plot_dict = {'RHS = Full Multiscale': {'color':'black', 'linestyle':'-'},
	# 			'RHS = Slow': {'color':'gray', 'linestyle':':'},
	# 			'Discrete Full': {'color':color_list[0], 'linestyle':'--'},

    eps_vec = [7, 5, 3, 1]

    eval_dict = {'t_valid_005': 'Validity Time', 'kl_all': 'KL-Divergence', 'acf_error': 'Autocorrelation Error'}
    fig, ax = plt.subplots(nrows=3, ncols=len(eps_vec), figsize=(14, 14), sharex='row', sharey='row')
    fig_path = os.path.join(plot_dir, 'l96_scale_compare_performance_trainNumber{}'.format(trainNumber))

    for col in range(len(eps_vec)):
        eps = eps_vec[col]
        ax[0][col].set_title('$\mathbf{{\\epsilon = 2^{{-{eps}}}}}$'.format(eps=eps), fontweight='bold', fontsize=22)
        output_dir = os.path.join(base_dir, 'l96ms_eps-{}_v0'.format(eps))
        full_summary_df_name = os.path.join(output_dir, 'full_summary_df.pickle')
        with open(full_summary_df_name, "rb") as file:
            summary_df = pickle.load(file)

        # add paper names
        summary_df['Model'] = summary_df.longname
        summary_df['Uses $f_0$'] = 'No'
        summary_df.loc[summary_df.usef0==1, 'Uses $f_0$'] = 'Yes'

        for nm in papername_dict:
            summary_df.loc[summary_df.longname==nm, 'Model'] = papername_dict[nm]

        paper_df = summary_df[(summary_df.longname.isin(papername_dict.keys())) &
                            (summary_df.stateType!='stateAndPred') &
                            (summary_df.fidelity==fid) &
                            (summary_df.dt==dt) &
                            (summary_df.tTrain==tTrain) &
                            (summary_df.eval_pickle_fname.str.contains("test_eval_{}.pickle".format(fid)))]

        # plot KDE
        keylist = ['kl_all', 'acf_error']
        for j in range(2):
            axfoo = ax[j][col]
            key = keylist[j]
            if key in ['kl_all', 't_valid_005', 't_valid_050']:
                xnm = "xgrid"
                y_true = "P_true"
                y_approx = "P_approx"
            else:
                xnm = "acf_grid"
                y_true = "acf_true"
                y_approx = "acf_approx"

            first = 1
            for nm in good_names: #papername_dict:
                try:
                    evalpath = paper_df[(paper_df.longname==nm) & (paper_df.trainNumber==trainNumber) & (paper_df.testNumber==0)].eval_pickle_fname[0]
                    dirnm = evalpath.split('/')[-2]
                    kdefile = os.path.join(output_dir, dirnm, 'kde_data_inv_meashifi.pickle')
                    with open(kdefile, "rb") as file:
                        kdedata = pickle.load(file)[0]
                except:
                    continue
                if first:
                    first = 0
                    axfoo.plot(kdedata[xnm], kdedata[y_true], label=papername_dict[nm], linewidth=1, color='black', linestyle='--')
                axfoo.plot(kdedata[xnm], kdedata[y_approx], label=papername_dict[nm], linewidth=1, **papername_styledict[nm])
                if key in ['kl_all']:
                    axfoo.set_xlim([-10, 15])
                axfoo.legend(loc='best', fontsize=8)
                if col==0:
                    axfoo.set_ylabel(ylabel_dict[key], fontsize=20)
                # axfoo.set_xlabel(r'$X_k$', fontsize=24)

        # ax[2][col] plot box plot
        key = 't_valid_005'
        axfoo = ax[2][col]
        xlabel = '$\\epsilon = 2^{{-{eps}}}$'.format(eps=eps)
        new_box(df=paper_df, fig_path=fig_path+'_small',
                    x="Model",
                    y=key,
                    order=[papername_dict[k] for k in papername_dict.keys()],
                    xlabel=None,
                    ylabel=None,
                    title=None,
                    legloc='best',
                    figsize=(12,12),
                    fontsize=10,
                    rotation=40,
                    ax=axfoo)
        if eval_dict[key]=='Validity Time':
            axfoo.set_yscale('linear')
        if col==0:
            axfoo.set_ylabel(eval_dict[key], fontsize=20)
        else:
            axfoo.set_ylabel(None)
        axfoo.set_xlabel(None)
        plt.savefig(fig_path)

    plt.savefig(fig_path, dpi=300)


    fig, ax = plt.subplots(nrows=2, ncols=len(eps_vec), figsize=(14, 7), sharex='row', sharey='row')
    eval_dict = {'kl_all': 'KL-Divergence', 'acf_error': 'Autocorrelation Error'}
    fig_path = os.path.join(plot_dir, 'l96_scale_compare_residuals_trainNumber{}'.format(trainNumber))

    for col in range(len(eps_vec)):
        eps = eps_vec[col]
        output_dir = os.path.join(base_dir, 'l96ms_eps-{}_v0'.format(eps))
        full_summary_df_name = os.path.join(output_dir, 'full_summary_df.pickle')
        with open(full_summary_df_name, "rb") as file:
            summary_df = pickle.load(file)

        # add paper names
        summary_df['Model'] = summary_df.longname
        summary_df['Uses $f_0$'] = 'No'
        summary_df.loc[summary_df.usef0==1, 'Uses $f_0$'] = 'Yes'

        for nm in papername_dict:
            summary_df.loc[summary_df.longname==nm, 'Model'] = papername_dict[nm]

        paper_df = summary_df[(summary_df.longname.isin(papername_dict.keys())) &
                            (summary_df.stateType!='stateAndPred') &
                            (summary_df.fidelity==fid) &
                            (summary_df.dt==dt) &
                            (summary_df.tTrain==tTrain) &
                            (summary_df.eval_pickle_fname.str.contains("test_eval_{}.pickle".format(fid)))]

        # ax[0][col] plot model fit
        axfoo = ax[0][col]
        first = 1
        ylim = None
        for nm in good_names[1:]: #papername_dict:
            try:
                evalpath = paper_df[(paper_df.longname==nm) & (paper_df.trainNumber==trainNumber) & (paper_df.testNumber==0)].eval_pickle_fname[0]
                dirnm = evalpath.split('/')[-2]
                datafile = os.path.join(output_dir, dirnm, 'training_fit_data.pickle')
                with open(datafile, "rb") as file:
                    data = pickle.load(file)[0]
                    			# data = [{'x_model': x_grid,
                    			# 		 'hY_model': hY,
                    			# 		 'x_data': x_input_descaled.reshape(-1),
                    			# 		 'hY_data': x_output.T.reshape(-1)}]

            except:
                continue
            if first:
                first = 0
                density_scatter(data["x_data"], data['hY_data'], ax = axfoo, n_subsample=100000, s=5)
                burnin = int(data["x_data"].shape[0] / 2)
                data_inds = burnin + np.arange(0,1*1000*9,9)
                ax[1][col].scatter(data["x_data"][data_inds], data['hY_data'][data_inds], s=3, c=plt.cm.cool(np.flip(np.linspace(0,1,len(data_inds)))), label='Trajectory (T=1)')
                ax[1][col].scatter(data["x_data"][data_inds[-1]], data['hY_data'][data_inds[-1]], s=130, marker='x', c=plt.cm.cool([0.]), label='m(t=0)')
                ax[1][col].scatter(data["x_data"][data_inds[0]], data['hY_data'][data_inds[0]], s=130, marker='x', c=plt.cm.cool([1.]), label='m(t=1)')
                # axfoo.scatter(data["x_data"], data['hY_data'], label='Approximate residuals', s=5, color='gray', alpha=0.8)
                ylim = axfoo.get_ylim()
            axfoo.plot(data["x_model"], data["hY_model"], label=papername_dict[nm], linewidth=2, **papername_styledict[nm])
            ax[1][col].plot(data["x_model"], data["hY_model"], label=papername_dict[nm], linewidth=2, **papername_styledict[nm])
        axfoo.set_ylim(ylim)
        ax[1][col].set_ylim(ylim)
        if col==0:
            axfoo.set_ylabel('$m$', fontsize=24, rotation=0)
            ax[1][col].set_ylabel('$m$', fontsize=24, rotation=0)
        axfoo.legend(loc='best', fontsize=8)
        ax[1][col].legend(loc='best', fontsize=8)
        axfoo.set_title('$\\epsilon = 2^{{-{eps}}}$'.format(eps=eps), fontweight='bold', fontsize=22)

        plt.savefig(fig_path)

    plt.savefig(fig_path, dpi=300)


def plot_l96eps(base_dir, plot_dir, experiment_dir='l96ms_ScaleSep_v0', trainNumber=0):

    os.makedirs(plot_dir, exist_ok=True)

    fid='hifi'
    dt=0.001
    tTrain=100
    # eval_dict = {'t_valid_005': 'Validity Time', 'kl_all': 'KL-Divergence', 'acf_error': 'Autocorrelation Error'}
    eval_dict = {'t_valid_050': 'Validity Time', 't_valid_005': 'Validity Time', 'acf_error': 'Autocorrelation Error', 'kl_all': 'KL-Divergence'}
    good_names = ['f0only resid=1, state', 'Psi CW resid=1, state', 'rhs w/ diff=Spline, costInt=datagrid CW resid=1, state']
    papername_dict = {'f0only resid=1, state': '$f^\dag \\approx f_0$',
                        'Psi resid=0, state': '$\Psi^\dag \\approx m$',
                        'rhs w/ diff=Spline, costInt=datagrid resid=0, state': '$f^\dag \\approx m$',
                        'Psi CW resid=1, state': '$\Psi^\dag \\approx \Psi_0 + m$',
                        'rhs w/ diff=Spline, costInt=datagrid CW resid=1, state': '$f^\dag \\approx f_0 + m$'
                        }
    papername_styledict = {'f0only resid=1, state': {'color': 'gray', 'linestyle': ':'},
                        'Psi resid=0, state': {'color': 'orange', 'linestyle': '--'},
                        'rhs w/ diff=Spline, costInt=datagrid resid=0, state': {'color': 'blue', 'linestyle': '--'},
                        'Psi CW resid=1, state': {'color': 'orange', 'linestyle': '-'},
                        'rhs w/ diff=Spline, costInt=datagrid CW resid=1, state': {'color': 'blue', 'linestyle': '-'}
                        }

    ylabel_dict = {'kl_all': 'Probability', 'acf_error': 'ACF', 't_valid_005': 'Validity Time', 't_valid_050': 'Validity Time'}
	# plot_dict = {'RHS = Full Multiscale': {'color':'black', 'linestyle':'-'},
	# 			'RHS = Slow': {'color':'gray', 'linestyle':':'},
	# 			'Discrete Full': {'color':color_list[0], 'linestyle':'--'},

    eps_vec = [7, 5, 3, 1]
    fig, ax = plt.subplots(nrows=3, ncols=len(eps_vec), figsize=(14, 14), sharex='row', sharey='row')
    first_key = True
    for key in eval_dict:
        fig_path = os.path.join(plot_dir, 'l96_scale_comparison_{}_trainNumber{}'.format(key, trainNumber))

        if key in ['kl_all', 't_valid_005', 't_valid_050']:
            xnm = "xgrid"
            y_true = "P_true"
            y_approx = "P_approx"
        else:
            xnm = "acf_grid"
            y_true = "acf_true"
            y_approx = "acf_approx"


        for col in range(len(eps_vec)):
            eps = eps_vec[col]
            output_dir = os.path.join(base_dir, 'l96ms_eps-{}_v0'.format(eps))
            full_summary_df_name = os.path.join(output_dir, 'full_summary_df.pickle')
            with open(full_summary_df_name, "rb") as file:
                summary_df = pickle.load(file)

            # add paper names
            summary_df['Model'] = summary_df.longname
            summary_df['Uses $f_0$'] = 'No'
            summary_df.loc[summary_df.usef0==1, 'Uses $f_0$'] = 'Yes'

            for nm in papername_dict:
                summary_df.loc[summary_df.longname==nm, 'Model'] = papername_dict[nm]

            paper_df = summary_df[(summary_df.longname.isin(papername_dict.keys())) &
                                (summary_df.stateType!='stateAndPred') &
                                (summary_df.fidelity==fid) &
                                (summary_df.dt==dt) &
                                (summary_df.tTrain==tTrain) &
                                (summary_df.eval_pickle_fname.str.contains("test_eval_{}.pickle".format(fid)))]

            # ax[0][col] plot model fit
            if first_key:
                axfoo = ax[0][col]
                first = 1
                ylim = None
                for nm in good_names[1:]: #papername_dict:
                    try:
                        evalpath = paper_df[(paper_df.longname==nm) & (paper_df.trainNumber==trainNumber) & (paper_df.testNumber==0)].eval_pickle_fname[0]
                        dirnm = evalpath.split('/')[-2]
                        datafile = os.path.join(output_dir, dirnm, 'training_fit_data.pickle')
                        with open(datafile, "rb") as file:
                            data = pickle.load(file)[0]
                            			# data = [{'x_model': x_grid,
                            			# 		 'hY_model': hY,
                            			# 		 'x_data': x_input_descaled.reshape(-1),
                            			# 		 'hY_data': x_output.T.reshape(-1)}]

                    except:
                        continue
                    if first:
                        first = 0
                        density_scatter(data["x_data"], data['hY_data'], ax = axfoo, n_subsample=100000, s=5)
                        axfoo.scatter(data["x_data"][-5000:], data['hY_data'][-5000:], s=3, color='red', alpha=0.1, label='T=5 trajectory')
                        # axfoo.scatter(data["x_data"], data['hY_data'], label='Approximate residuals', s=5, color='gray', alpha=0.8)
                        ylim = axfoo.get_ylim()
                    axfoo.plot(data["x_model"], data["hY_model"], label=papername_dict[nm], linewidth=2, **papername_styledict[nm])
                axfoo.set_ylim(ylim)
                if col==0:
                    axfoo.set_ylabel('$\\hat{m}$')
                axfoo.legend(loc='best', fontsize=8)
                axfoo.set_title('$\\epsilon = 2^{{-{eps}}}$'.format(eps=eps), fontweight='bold', fontsize=22)


            # ax[1][col] plot KDE/ACF
            axfoo = ax[1][col]
            axfoo.clear()
            first = 1
            for nm in good_names: #papername_dict:
                try:
                    evalpath = paper_df[(paper_df.longname==nm) & (paper_df.trainNumber==trainNumber) & (paper_df.testNumber==0)].eval_pickle_fname[0]
                    dirnm = evalpath.split('/')[-2]
                    kdefile = os.path.join(output_dir, dirnm, 'kde_data_inv_meashifi.pickle')
                    with open(kdefile, "rb") as file:
                        kdedata = pickle.load(file)[0]
                except:
                    continue
                if first:
                    first = 0
                    axfoo.plot(kdedata[xnm], kdedata[y_true], label=papername_dict[nm], linewidth=1, color='black', linestyle='--')
                axfoo.plot(kdedata[xnm], kdedata[y_approx], label=papername_dict[nm], linewidth=1, **papername_styledict[nm])
            if key in ['kl_all', 't_valid_005', 't_valid_050']:
                axfoo.set_xlim([-10, 15])
            axfoo.legend(loc='best', fontsize=8)
            if col==0:
                axfoo.set_ylabel(ylabel_dict[key])

            # ax[2][col] plot box plot
            axfoo = ax[2][col]
            axfoo.clear()
            xlabel = '$\\epsilon = 2^{{-{eps}}}$'.format(eps=eps)
            new_box(df=paper_df, fig_path=fig_path+'_small',
                        x="Model",
                        y=key,
                        order=[papername_dict[k] for k in papername_dict.keys()],
                        xlabel=None,
                        ylabel=eval_dict[key],
                        title=eval_dict[key],
                        legloc='best',
                        figsize=(12,12),
                        fontsize=10,
                        rotation=40,
                        ax=axfoo)
            if eval_dict[key]=='Validity Time':
                axfoo.set_yscale('linear')
            # ax[2][col].set_xlabel(xlabel, fontsize=16, fontweight='bold')

            plt.savefig(fig_path)

        first_key = False
        plt.savefig(fig_path, dpi=300)



####### L96MS EXPERIMENT
def plot_l96ScaleSep(base_dir, plot_dir, experiment_dir='l96ms_ScaleSep_v0'):
    os.makedirs(plot_dir, exist_ok=True)
    output_dir = os.path.join(base_dir, experiment_dir)

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
    for trainNumber in range(10):
        plot_l96eps_v2(**FLAGS.__dict__, trainNumber=trainNumber)
        plot_l96eps(**FLAGS.__dict__, trainNumber=trainNumber)
    plot_l96ScaleSep(base_dir=FLAGS.base_dir, plot_dir='/Users/matthewlevine/Dropbox/mechanistic+ML/active_writeups/2021_ICML/l96_figs/auto_eps-5', experiment_dir='l96ms_eps-5_v0')
    plot_l96ScaleSep(base_dir=FLAGS.base_dir, plot_dir='/Users/matthewlevine/Dropbox/mechanistic+ML/active_writeups/2021_ICML/l96_figs/auto_eps-3', experiment_dir='l96ms_eps-3_v0')
    plot_l96ScaleSep(base_dir=FLAGS.base_dir, plot_dir='/Users/matthewlevine/Dropbox/mechanistic+ML/active_writeups/2021_ICML/l96_figs/auto_eps-1', experiment_dir='l96ms_eps-1_v0')
    plot_l96ScaleSep(**FLAGS.__dict__)
