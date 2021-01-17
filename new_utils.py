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
from new_utils import *
import pdb


def queue_joblist(combined_settings, shared_settings, output_dir, master_job_file, cmd):
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
            master_job_file=master_job_file, no_submit=True)

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
                    hours=6, no_submit=False, use_gpu=False):

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
    if exclusive:
        sbatch_str += "#SBATCH --exclusive\n" # exclusive use of a node for the submitted job
    if use_gpu:
        sbatch_str += "#SBATCH --gres=gpu:1\n" # use a GPU for the computation
    sbatch_str += module_load_command
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
            if float(val)==int(val):
                val = int(val)
            else:
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
    			n_test_traj,
    			n_train_traj,
    			delta_t,
                data_pathname):

    f_ode = lambda t, y: ode.rhs(y,t)

    def simulate_traj(T1, T2):
        t0 = 0
        u0 = ode.get_inits()
        print("Initial transients...")
        tstart = time()
        t_span = [t0, T1]
        t_eval = np.array([t0+T1])
        sol = solve_ivp(fun=f_ode, t_span=t_span, y0=u0, t_eval=t_eval, max_step=delta_t, method='RK45')
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
        sol = solve_ivp(fun=f_ode, t_span=t_span, y0=u0, t_eval=t_eval, max_step=delta_t, method='RK45')
        u = sol.y.T
        print('took', '{:.2f}'.format((time() - tstart)/60),'minutes')
        return u

    # make 1 long inv-meas trajectory
    u_inv_meas = simulate_traj(T1=t_transient, T2=t_invariant_measure)

    # make many training trajectories
    u_train = np.array([simulate_traj(T1=t_transient, T2=t_train) for _ in range(n_train_traj)])

    # make many testing trajectories
    u_test = np.array([simulate_traj(T1=t_transient, T2=t_test) for _ in range(n_test_traj)])

    # save data
    data = {
        "u_inv_meas": u_inv_meas,
        "u_train": u_train,
        "u_test": u_test,
        "dt": delta_t
    }

    os.makedirs(os.path.dirname(data_pathname), exist_ok=True)
    with open(data_pathname, "wb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
