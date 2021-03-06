import os, sys
import numpy as np
import argparse
from time import time
from meta_model import *
from utils import *
import pdb
from pydoc import locate

parser = argparse.ArgumentParser()
parser.add_argument('--settings_path', type=str)
FLAGS = parser.parse_args()


# main function
def main():
    settings = get_settings(settings_path=FLAGS.settings_path)
    model = IDK(settings=settings)

    # if the model has already been trained, load it.
    # else, train it
    new_run = 1
    if model.retrain:
        print('Begin training!')
        model.train()
        model.loadModel() # need to do this because last validation run needs to be cleared
    else:
        try:
            model.loadModel()
            print('Skip training!')
            new_run = 0
        except:
            print('Begin training!')
            model.train()
            model.loadModel() # need to do this because last validation run needs to be cleared

    # test the saved model
    if new_run or model.retest:
        print('Begin testing!')
        model.test()


def get_settings(settings_path):
    # read in settings
    settings = file_to_dict(settings_path)
    saving_path = os.path.dirname(settings_path)

    if "experimentType" not in settings:
        settings["experimentType"] = 'pathak'

    # add settings
    settings["saving_path"] = saving_path
    settings["train_data_path"] = settings["data_pathname"]
    settings["test_data_path"] = settings["data_pathname"]
    settings["delta_t"] = settings["dt"]
    ODE = locate('odelibrary.{}'.format(settings["f0_name"]))
    if settings['f0_name']=='L96M':
        if settings["usef0"]:
            physics = ODE(slow_only=settings['slowOnly'])
            settings["f0"] = lambda t, y: physics.rhs(y, t)
        if settings["diff"]=="TrueDeriv":
            raise ValueError("True Derivatives not yet setup for L96MS case")
    else:
        if settings["usef0"]:
            if settings["experimentType"]=='pathak':
                physics1 = ODE()
                eps = settings['f0eps']
                physics1.b = physics1.b*(1+eps)
            elif settings["experimentType"]=='GPerror':
                physics1 = ODE(random_closure=True, epsGP=settings['f0eps'])
            elif settings["experimentType"]=='wrongModel':
                physics1 = ODE()
            settings["f0"] = lambda t, y: physics1.rhs(y, t)

        if settings["diff"]=="TrueDeriv":
            physics2 = ODE()
            settings["fTRUE"] = lambda t, y: physics2.rhs(y, t)
    return settings

if __name__ == '__main__':
    start_time = time()
    main()
    total_time = time() - start_time
    print("Total run time is {:2.2f} minutes".format(total_time/60))
