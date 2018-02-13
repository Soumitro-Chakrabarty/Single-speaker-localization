from __future__ import absolute_import
from __future__ import print_function
import numpy as np
#from git import Repo
import importers
import configargparse
import shutil
import keras
import uuid
import time
import json
import os
import scipy.io
#Additional imports from keras for Test File
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop,Adagrad
from keras.utils import np_utils
from keras.layers import TimeDistributed, Masking
#from load_train_data import load_input_train_data
from keras.models import model_from_json, load_model
#import matplotlib.pyplot as plt
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ModelCheckpoint
from keras import backend as K

K.set_image_dim_ordering('th')

def test(
    dataset_name,
    model_name,
    model_weights,
    optimizer="adam",
    objective="binary_crossentropy",
    plot_io=False,
    data_path='data/',
    data_cache_path=None,
    base_out_path='results/',
    verbose=2,
    seed=42
):

    dataset_file = importers.dataset_file(
        data_path, dataset_name
    )



    data = importers.Dataset(
        dataset_file,
    )


    # just reshape for print
    # load json and create model
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weights + '.h5')
    print("Loaded model from disk")

    test_it = data.TrainingGenerator(
        shuffle=False,
        seed=42,
        dataset='_test',
    )

    X, Y = data.load_data(index=1, dataset='_test')

    # Define the optimizer
    lrate = 0.01
    adam = Adam(lr =lrate,beta_1=0.9,beta_2=0.999,epsilon=1e-08)
    loaded_model.compile(loss=objective, optimizer= optimizer)
    # predict data based on test
    y_pred = loaded_model.predict_generator(
        test_it,
        val_samples=data.samples_per_epoch_test,
        max_q_size=100
    )

    return data, y_pred


if __name__ == '__main__':

    parser = configargparse.ArgumentParser(description='doa')
    parser.add_argument(
        '--confile',
        required=True,
        is_config_file=True, 
        help='config file path'
    )
    parser.add_argument(
        '--network',
        help='the name of the network'
    )
    parser.add_argument(
        '--output_path',
        help='results_path',
        default="results"
    )
    parser.add_argument(
        '--matfile',
        help='name of mat file',
        default="current_results"
    )
    parser.add_argument(
        '--weights',
        help='file with weights',
        default="results/weights"
    )
    parser.add_argument(
        '--optimizer',
        help='Set the optimizer',
        type=str,
        default="adam"
    )
    parser.add_argument(
        '--objective',
        help='string with the loss function',
        type=str,
        default="categorical_crossentropy"
    )
    parser.add_argument(
        '--dataset',
        help='Set the dataset',
        type=str,
        default="CNN_noisetraining_MCT_trial.hdf5"
    )
    parser.add_argument(
        '--data_path',
        help='path to dataset',
        type=str,
        default='data/'
    )
    parser.add_argument(
        '--data_cache_path',
        help='cache dataset here before running',
        type=str,
        default='/tmp'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        default=False,
        help='plot_data'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42
    )
    parser.add_argument(
        '--verbose',
        help='0 for no logging to stdout, 1 for progress bar logging, \
              2 for one log line per epoch.',
        type=int,
        default=1
    )
    parser.add_argument(
        '--uuid',
        help='experiment_id',
        nargs='?',
        type=str,
    )

    args = parser.parse_args()

data, y_pred = test(
    dataset_name=args.dataset,
    model_name=args.network,
    model_weights=args.weights,
    optimizer=args.optimizer,
    objective=args.objective,
    plot_io=args.plot,
    data_path=args.data_path,
    data_cache_path=args.data_cache_path,
    base_out_path=args.output_path,
    verbose=args.verbose,
    seed=args.seed
)

# write predictions to file
scipy.io.savemat(args.matfile + '.mat', mdict={'Output': y_pred})
