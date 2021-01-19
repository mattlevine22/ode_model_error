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

font = {'size': 16}
matplotlib.rc('font', **font)

import pdb

def summarize_eps(df, hue, style, output_dir, metric_list):
    for metric in metric_list:
        fig_path = os.path.join(output_dir, 'summary_eps_{}'.format(metric))
        fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(12, 12))
        sns.lineplot(ax=ax, data=df, x="f0eps", y=metric, style=style, hue=hue, err_style='bars')
        plt.savefig(fig_path)
        plt.close()

def plotMatrixSpectrum(model, A, mat_name):
    fig_path = model.saving_path + model.fig_dir + model.model_name + "/singular_values_{:}.png".format(mat_name)
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
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/eigenvalues_{:}.png".format(mat_name)
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
    fig_path = model.saving_path + model.fig_dir + "/matrix_{:}.png".format(mat_name)
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
