import os, sys
from time import time
from datetime import timedelta
import numpy as np
import pickle
import json
from scipy.integrate import solve_ivp
import subprocess
import glob
import itertools
import pandas as pd
from plotting_utils import *
import pdb

def optimizer_as_df(optimizer):
    opt_list = []
    for el in optimizer.res:
        new_dict = el['params']
        new_dict['target'] = el['target']
        opt_list.append(new_dict)
    df = pd.DataFrame(opt_list)
    return df

def df_eval(df):
    # read in things
    df_list = []
    for i in range(len(df)):
        test_fname = df.eval_pickle_fname.iloc[i]
        model_fname = df.model_fname.iloc[i]
        try:
            with open(test_fname, "rb") as file:
                data = pickle.load(file)
            data['eval_pickle_fname'] = test_fname
            data['testNumber'] = data.index
            # now add hyperparam info
            try:
                with open(model_fname, "rb") as file:
                    model = pickle.load(file)
                data['regularization_RF'] = model['regularization_RF']
                data['rf_Win_bound'] = model['rf_Win_bound']
                data['rf_bias_bound'] = model['rf_bias_bound']
                data['differentiation_error'] = model['differentiation_error']
            except:
                pass
            sub_df = pd.merge(df, data, on='eval_pickle_fname')
            df_list.append(sub_df)
        except:
            pass
    final_df = pd.concat(df_list)
    return final_df

def prioritized_job_sender(all_job_fnames, bash_command, list_of_priorities, do_all=False, noredo=True):

    if do_all:
        list_of_priorities.append(['']) # this includes all remaining things at the end

    for check_list in list_of_priorities:
        rmv_nms = []
        for job_fname in all_job_fnames:
            eval_nm = os.path.join(os.path.dirname(job_fname), 'test_eval.pickle')
            if noredo and os.path.isfile(eval_nm):
                rmv_nms.append(job_fname)
            elif all(elem in job_fname for elem in check_list):
                rmv_nms.append(job_fname)
                submit_job(job_fname, bash_command=bash_command)
        [all_job_fnames.remove(nm) for nm in rmv_nms]

    return

def queue_joblist(combined_settings, shared_settings, output_dir, master_job_file, cmd, hours=12, conda_env=None):
    nm_list = combined_settings.keys()
    all_settings_list = dict_combiner(combined_settings)

    # write JSON settings files and job files
    job_fname_list = []
    for settings_dict in all_settings_list:
        settings_dict.update(shared_settings)
        # write structured JSON file with run settings
        settings_path = make_output_path(settings_dict, output_dir=output_dir, suff='.json', nm_list=nm_list)
        dict_to_file(mydict=settings_dict, fname=settings_path)

        # write a Job file that points to the JSON settings file
        command_flag_dict = {'settings_path': settings_path}
        jobfile_dir = os.path.dirname(settings_path)
        jobstatus, jobnum, jobfile_path = make_and_deploy(bash_run_command=cmd,
            command_flag_dict=command_flag_dict,
            jobfile_dir=jobfile_dir,
            master_job_file=master_job_file, no_submit=True,
            conda_env=conda_env,
            hours=hours)

        job_fname_list.append(jobfile_path)
    return job_fname_list


def submit_job(job_filename, bash_command='zsh'):
    cmd = [bash_command, job_filename]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # check for successful run and print the error
    status = proc.returncode
    if status!=0:
        print('Job submission FAILED:', proc.stdout, cmd)
    else:
        print('Job submitted:', ' '.join(cmd))
    return status

def make_and_deploy(bash_run_command='echo $HOME',
                    module_load_command='module load python3/3.7.0\n',
                    command_flag_dict={}, jobfile_dir='./my_jobs',
                    jobname='jobbie', depending_jobs=[], jobid_dir=None,
                    master_job_file=None, report_status=True, exclusive=True,
                    hours=12, no_submit=False, use_gpu=False, conda_env=None,
                    mem=None):

    # build sbatch job script and write to file
    job_directory = jobfile_dir
    out_directory = jobfile_dir
    # job_directory = os.path.join(jobfile_dir,'.job')
    # out_directory = os.path.join(jobfile_dir,'.out')
    # os.makedirs(job_directory, exist_ok=True)
    # os.makedirs(out_directory, exist_ok=True)


    job_file = os.path.join(job_directory,"{0}.job".format(jobname))

    sbatch_str = ""
    sbatch_str += "#!/bin/bash\n"
    sbatch_str += "#SBATCH --account=astuart\n" # account name
    sbatch_str += "#SBATCH --job-name=%s.job\n" % jobname
    sbatch_str += "#SBATCH --output=%s.out\n" % os.path.join(out_directory,jobname)
    sbatch_str += "#SBATCH --error=%s.err\n" % os.path.join(out_directory,jobname)
    sbatch_str += "#SBATCH --time={0}:00:00\n".format(hours) # default 24hrs. Shorter time gets more priority.
    if mem:
        sbatch_str += "#SBATCH --mem={0}\n".format(mem) # amt of RAM per job (e.g. '10GB' or '100MB')
    if exclusive:
        sbatch_str += "#SBATCH --exclusive\n" # exclusive use of a node for the submitted job
    if use_gpu:
        sbatch_str += "#SBATCH --gres=gpu:1\n" # use a GPU for the computation
    sbatch_str += module_load_command
    if conda_env:
        sbatch_str += 'conda activate {}'.format(conda_env)
    sbatch_str += bash_run_command
    # sbatch_str += "python $HOME/mechRNN/experiments/scripts/run_fits.py"
    for key in command_flag_dict:
        sbatch_str += ' --{0} {1}'.format(key, command_flag_dict[key])
	# sbatch_str += ' --output_path %s\n' % experiment_dir

    sbatch_str += '\n'
    with open(job_file, 'w') as fh:
        fh.writelines(sbatch_str)

    # run the sbatch job script
    depending_jobs = [z for z in depending_jobs if z is not None]
    cmd = ['sbatch']
    if depending_jobs:
        depstr = ','.join(depending_jobs) #depending_jobs must be list of strings
        cmd.append('--dependency=after:{0}'.format(depstr))

    cmd.append(job_file)

    if no_submit:
        print('Job created (not submitted):', ' '.join(cmd))
        return 0, None, job_file

    proc = subprocess.run(cmd, capture_output=True, text=True)
    # check for successful run and print the error
    status = proc.returncode
    if report_status:
        if status!=0:
            print('Job submission FAILED:', proc.stdout, cmd)
        else:
            print('Job submitted:', ' '.join(cmd))

    jobnum = proc.stdout.strip().split(' ')[-1]
    if master_job_file:
        with open(master_job_file, 'a') as f:
            f.write('{0},{1}\n'.format(jobnum, ' '.join(cmd)))

    if jobid_dir:
        # write job_id to its target directory for easy checking later
        with open(os.path.join(jobid_dir,'{0}.id'.format(jobnum)), 'w') as fp:
            pass

    return status, jobnum, job_file

def dict_combiner(mydict):
    if mydict:
        keys, values = zip(*mydict.items())
        experiment_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    else:
        experiment_list = [{}]
    return experiment_list

def dict_to_file(mydict, fname):
	with open(fname, 'w') as f:
		json.dump(mydict, f, indent=2)
	return

def file_to_dict(fname):
    with open(fname) as f:
        my_dict = json.load(f)
    return my_dict

def make_output_path(setts, output_dir, nm_list, suff=''):
    nm_list = ['{key}-{val}'.format(key=key, val=setts[key]) for key in nm_list]
    nm_list.sort()
    model_path = os.path.join(output_dir, '_'.join(nm_list))
    os.makedirs(model_path, exist_ok=True)
    pathname = os.path.join(model_path, 'settings' + suff)
    return pathname

def parse_output_path(pathname, nm_list):
    foo = os.path.basename(pathname).split('_')
    var_dict = {}
    for s in foo:
        ss = s.split('-')
        key = ss[0]
        try:
            val = ss[1]
        except:
            pdb.set_trace()
        try:
            val = int(val)
        except:
            try:
                val = float(val)
            except:
                pass
        var_dict[key] = val
    return var_dict


def generate_data(ode,
                rng_seed,
    			t_transient,
    			t_train,
    			t_invariant_measure,
    			t_test,
                t_validate,
    			n_test_traj,
    			n_train_traj,
                n_validate_traj,
    			delta_t,
                data_pathname,
                solver_type):

    # load solver dict
    solver_dict='./Config/solver_settings.json'
    foo = file_to_dict(solver_dict)
    solver_settings = foo[solver_type]


    f_ode = lambda t, y: ode.rhs(y,t)

    def simulate_traj(T1, T2):
        t0 = 0
        u0 = ode.get_inits()
        print("Initial transients...")
        tstart = time()
        t_span = [t0, T1]
        t_eval = np.array([t0+T1])
        sol = solve_ivp(fun=f_ode, t_span=t_span, y0=u0, t_eval=t_eval, **solver_settings)
        print('took', '{:.2f}'.format((time() - tstart)/60),'minutes')

        print("Integration...")
        tstart = time()
        u0 = np.squeeze(sol.y)
        t_span = [t0, T2]
        t_eval_tmp = np.arange(t0, T2, delta_t)
        t_eval = np.zeros(len(t_eval_tmp)+1)
        t_eval[:-1] = t_eval_tmp
        t_eval[-1] = T2
        # sol = solve_ivp(fun=lambda t, y: self.rhs(t0, y0), t_span=t_span, y0=u0, method=testcontinuous_ode_int_method, rtol=testcontinuous_ode_int_rtol, atol=testcontinuous_ode_int_atol, max_step=testcontinuous_ode_int_max_step, t_eval=t_eval)
        sol = solve_ivp(fun=f_ode, t_span=t_span, y0=u0, t_eval=t_eval, **solver_settings)
        u = sol.y.T
        print('took', '{:.2f}'.format((time() - tstart)/60),'minutes')
        return u

    try:
        ode.set_random_predictor()
        gpr_predict = ode.predictor
        figdir = os.path.join(os.path.dirname(data_pathname), 'GPFigures')
        # plot_io_characteristics(figdir=figdir, X=ode.X, y=ode.y)
        figdir = os.path.join(os.path.dirname(data_pathname), 'GPFigures2')
        plot_io_characteristics(figdir=figdir, X=ode.Xdense, gpr_predict=gpr_predict)
    except:
        gpr_predict = lambda x: 0

    # make 1 long inv-meas trajectory
    u_inv_meas = np.array([simulate_traj(T1=t_transient, T2=t_invariant_measure) for _ in range(1)])

    figdir = os.path.join(os.path.dirname(data_pathname), 'DataFigures')
    plot_model_characteristics(figdir=figdir, X=u_inv_meas)

    # make many training trajectories
    u_train = np.array([simulate_traj(T1=t_transient, T2=t_train) for _ in range(n_train_traj)])
    udot_train = np.array([[f_ode(0, u_train[k,j]) for j in range(u_train.shape[1])] for k in range(n_train_traj)])

    # make many testing trajectories
    u_test = np.array([simulate_traj(T1=t_transient, T2=t_test) for _ in range(n_test_traj)])
    udot_test = np.array([[f_ode(0, u_test[k,j]) for j in range(u_test.shape[1])] for k in range(n_test_traj)])

    # make many validation trajectories
    u_validate = np.array([simulate_traj(T1=t_transient, T2=t_validate) for _ in range(n_validate_traj)])
    udot_validate = np.array([[f_ode(0, u_validate[k,j]) for j in range(u_validate.shape[1])] for k in range(n_validate_traj)])

    # save data
    data = {
        "u_inv_meas": u_inv_meas,
        "u_train": u_train,
        "u_test": u_test,
        "u_validate": u_validate,
        "udot_train": udot_train,
        "udot_test": udot_test,
        "udot_validate": udot_validate,
        "dt": delta_t,
        "gpr_error": gpr_predict
    }

    os.makedirs(os.path.dirname(data_pathname), exist_ok=True)
    with open(data_pathname, "wb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
