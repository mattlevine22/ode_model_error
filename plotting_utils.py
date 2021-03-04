#!/usr/bin/env python
import os
import numpy as np
from scipy.linalg import svdvals, eigvals
from scipy.sparse.linalg import svds as sparse_svds
from scipy.sparse.linalg import eigs as sparse_eigs


# Plotting parameters
import matplotlib
import pandas as pd
matplotlib.use("Agg")

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib  import cm
import pickle
from matplotlib import colors
import six
color_dict = dict(six.iteritems(colors.cnames))

# font = {'size': 16}
# matplotlib.rc('font', **font)

# sns.set(rc={'text.usetex': True}, font_scale=4)


import pdb

def box(df, output_dir, metric_list, x="model_name", fname_shape='summary_eps_{}', figsize=(24, 20)):
    for metric in metric_list:
        try:
            fig_path = os.path.join(output_dir, fname_shape.format(metric))
            fig, ax = plt.subplots(nrows=1, ncols=1,figsize=figsize)
            sns.boxplot(ax=ax, data=df, x=x, y=metric)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=20, horizontalalignment='right', fontsize='large')

            plt.savefig(fig_path)
            ax.set_yscale('log')
            plt.savefig(fig_path + '_ylog')

        except:
            print('Failed at', metric)
            pass
        plt.close()

def new_box(df, fig_path,
            x='Model',
            y='t_valid_005',
            order=None,
            figsize=(12,10),
            fontsize=20,
            ylabel=None,
            xlabel=None,
            title=None,
            legloc='upper right'):

    font = {'size': fontsize}
    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=figsize)
    sns.boxplot(ax=ax, data=df, x=x, y=y, order=order)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, horizontalalignment='right', fontsize='large')
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontsize=fontsize)
    # ax.legend(loc=legloc, fontsize=fontsize)

    # fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.subplots_adjust(bottom=0.15, left=0.15)


    plt.savefig(fig_path)
    ax.set_yscale('log')
    plt.savefig(fig_path + '_ylog')
    plt.close()


def new_summary(df, fig_path, hue='Model', style='Uses $f_0$', x="$\epsilon$", y='t_valid_005',
                figsize=(12,10),
                fontsize=20,
                ylabel=None,
                xlabel=None,
                title=None,
                estimator=np.median,
                ci='sd',
                legloc='upper right'):

    font = {'size': fontsize}
    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    sns.lineplot(ax=ax, data=df, estimator=estimator, ci=ci, x=x, y=y, style=style, hue=hue, err_style='bars', linewidth=4)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontsize=fontsize)
    ax.legend(loc=legloc, fontsize=fontsize)
    plt.savefig(fig_path)

    ax.set_yscale('log')
    plt.savefig(fig_path + '_ylog')

    ax.set_xscale('log')
    plt.savefig(fig_path + '_xlog_ylog')

    ax.set_yscale('linear')
    plt.savefig(fig_path + '_xlog')

    plt.close()


def summarize(df, hue, style, output_dir, metric_list, x="f0eps", fname_shape='summary_eps_{}', figsize=(24, 12)):
    for metric in metric_list:
        try:
            fig, ax = plt.subplots(nrows=1, ncols=1,figsize=figsize)
            sns.lineplot(ax=ax, data=df, x=x, y=metric, style=style, hue=hue, err_style='bars', linewidth=4, ci='sd')
            fig_path = os.path.join(output_dir, fname_shape.format(metric))
            plt.savefig(fig_path)
            ax.set_yscale('log')
            plt.savefig(fig_path + '_ylog')

            ax.set_xscale('log')
            plt.savefig(fig_path + '_xlog_ylog')

            ax.set_yscale('linear')
            plt.savefig(fig_path + '_xlog')
        except:
            print('Failed at', metric)
            pass
        plt.close()

def plotMatrixSpectrum(model, A, mat_name):
    fig_path = os.path.join(model.fig_dir, "singular_values_{:}.png".format(mat_name))
    try:
        s = svdvals(A)
    except:
        s = -np.sort(-sparse_svds(A, return_singular_vectors=False, k=min(100,min(A.shape))))
    plt.plot(s,'o')
    plt.ylabel(r'$\sigma$')
    plt.title('Singular values of {:}'.format(mat_name))
    plt.savefig(fig_path)
    plt.close()

    if A.shape[0]==A.shape[1]: #is square
        fig_path = os.path.join(model.fig_dir, "eigenvalues_{:}.png".format(mat_name))
        try:
            eig = eigvals(A)
        except:
            eig = sparse_eigs(A, return_eigenvectors=False, k=min(1000,min(A.shape)))
        plt.plot(eig.real, eig.imag, 'o')
        plt.xlabel(r'Re($\lambda$)')
        plt.ylabel(r'Im($\lambda$)')
        plt.title('Eigenvalues of {:}'.format(mat_name))
        plt.savefig(fig_path)
        plt.close()

def plotMatrix(model, A, mat_name):
    fig_path = os.path.join(model.fig_dir, "matrix_{:}.png".format(mat_name))
    # plot matrix visualizations
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(12, 6))
    foo = ax[0].matshow(A, vmin=np.min(A), vmax=np.max(A), aspect='auto')
    # ax[0].axes.xaxis.set_visible(False)
    # ax[0].axes.yaxis.set_visible(False)
    ax[0].set_title(mat_name)
    fig.colorbar(foo, ax=ax[0])

    sns.ecdfplot(data=np.abs(A).reshape(-1,1), ax=ax[1])
    ax[1].set_xscale('log')
    ax[1].set_title('Distribution of matrix entries')
    plt.savefig(fig_path)
    plt.close()


def plot_model_characteristics(figdir, X, fontsize=20):
    os.makedirs(figdir, exist_ok=True)
    font = {'size': fontsize}
    matplotlib.rc('font', **font)

    X = np.squeeze(X)
    # Pool data and plot Inv Measure
    fig_path = os.path.join(figdir, "inv_stats_POOL.png")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 14))
    sns.kdeplot(X.reshape(-1), ax=ax, linewidth=4)
    plt.savefig(fig_path)
    plt.close()

    # Plot Invariant Measure of each state individually as a marginal
    fig_path = os.path.join(figdir, "inv_stats_MARGINAL.png")
    ndim = X.shape[1]
    fig, ax = plt.subplots(nrows=1, ncols=ndim, figsize=(24, 14))
    for x_in in range(ndim): #
        sns.kdeplot(X[:,x_in], ax=ax[x_in], linewidth=4)
        ax[x_in].set_xlabel('X_{}'.format(x_in))
    plt.savefig(fig_path)
    plt.close()

    # Plot Invariant Measure of each state individually, plus bivariate scatter
    fig_path = os.path.join(figdir, "inv_stats_BIVARIATE.png")
    ndim = X.shape[1]
    fig, ax = plt.subplots(nrows=ndim, ncols=ndim, figsize=(14, 14))
    # x axis is INPUT dim for model
    # y axis is OUTPUT dim for model
    for x_in in range(ndim): #
        for x_out in range(ndim):
            if x_out==x_in:
                sns.kdeplot(X[:,x_in], ax=ax[x_out][x_in], linewidth=4)
                # ax[x_out][x_in].ksdensity(X[:,x_in])
            else:
                # sns.scatter(x=X[:,x_in], y=X[:,x_out], ax=ax[x_out][x_in])
                ax[x_out][x_in].scatter(X[:,x_in], X[:,x_out])
            ax[x_out][x_in].set_xlabel('X_{}'.format(x_in))
            ax[x_out][x_in].set_ylabel('X_{}'.format(x_out))
    plt.savefig(fig_path)
    plt.close()


def plot_io_characteristics(figdir, X, y=None, gpr_predict=None, fontsize=20):

    if y is None:
        y = gpr_predict(X)

    os.makedirs(figdir, exist_ok=True)
    font = {'size': fontsize}
    matplotlib.rc('font', **font)

    # Plot bivariate scatter
    fig_path = os.path.join(figdir, "bivariate_scatter.png")
    ndim = X.shape[1]
    fig, ax = plt.subplots(nrows=ndim, ncols=ndim, figsize=(14, 14))
    # x axis is INPUT dim for model
    # y axis is OUTPUT dim for model
    for x_in in range(ndim): #
        for x_out in range(ndim):
            ax[x_out][x_in].scatter(X[:,x_in], y[:,x_out])
            ax[x_out][x_in].set_xlabel('Xin_{}'.format(x_in))
            ax[x_out][x_in].set_ylabel('Yout_{}'.format(x_out))
    plt.savefig(fig_path)
    plt.close()

    # Plot fancier thing
    for y_out in range(ndim):
        fig_path = os.path.join(figdir, "contour_{}.png".format(y_out))
        fig, ax = plt.subplots(nrows=ndim, ncols=ndim, figsize=(14, 14))
        # x axis is INPUT dim for model
        # y axis is OUTPUT dim for model
        for x_in1 in range(ndim): #
            for x_in2 in range(ndim):
                # xxin1, xxin2 = np.meshgrid(X[x_in1], X[_in2])
                # ax[x_in2][x_in1].contourf(xxin1, xxin2, np.squeeze(y[:,y_out]))
                ax[x_in2][x_in1].scatter(x=X[:,x_in1], y=X[:,x_in2], c=np.squeeze(y[:,y_out]), alpha=0.5)
                ax[x_in2][x_in1].set_xlabel('Xin_{}'.format(x_in1))
                ax[x_in2][x_in1].set_ylabel('Xin_{}'.format(x_in2))
        fig.suptitle('GP field for output state {}'.format(y_out))
        plt.savefig(fig_path)
        plt.close()
