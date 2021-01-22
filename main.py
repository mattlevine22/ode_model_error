import os, sys
import numpy as np
import argparse
from meta_model import *
from utils import *
import pdb

from odelibrary import L63
parser = argparse.ArgumentParser()
parser.add_argument('--settings_path', type=str)
FLAGS = parser.parse_args()


# main function
def main():
    settings = get_settings(settings_path=FLAGS.settings_path)
    model = IDK(settings=settings)

    # if the model has already been trained, load it.
    # else, train it
    try:
        model.loadModel()
    except:
        print('Begin training!')
        model.train()
        model.loadModel() # need to do this because last validation run needs to be cleared

    # test the saved model
    print('Begin testing!')
    model.test()

    # plot the model performance
    model.plot()

def get_settings(settings_path):
    # read in settings
    settings = file_to_dict(settings_path)
    saving_path = os.path.dirname(settings_path)

    # add settings
    settings["saving_path"] = saving_path
    settings["train_data_path"] = settings["data_pathname"]
    settings["test_data_path"] = settings["data_pathname"]
    settings["delta_t"] = settings["dt"]
    if settings["usef0"]:
        physics = L63()
        eps = settings['f0eps']
        physics.b = physics.b*(1+eps)
        settings["f0"] = lambda t, y: physics.rhs(y, t)

    return settings

if __name__ == '__main__':
	main()
