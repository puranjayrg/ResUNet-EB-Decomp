#--------
# Imports
#--------
import matplotlib
from tensorflow.python.keras.callbacks import ModelCheckpoint
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

import os
import sys

import healpy as hp
import scipy as sc

import h5py
import pathlib
import segmentation_models as sm
import keras_tuner as kt

keras.backend.set_image_data_format('channels_last')

#-------------
# Data Loading
#-------------
res_path = pathlib.Path("/home/puranjay/projects/def-hinshaw/puranjay/results/resunet_realb/sm/run11_11-03-22")
data_path = pathlib.Path("/home/puranjay/projects/def-hinshaw/puranjay/data")
res_path.mkdir(parents=True, exist_ok=True)

with h5py.File(os.path.join(data_path, 'traindata_1patch20deg_ns1024_12k5.h5'), 'r') as f:
    print('Opened h5 file, now getting data...', flush=True)
    dset_xtrain = f.get('train/x_train')
    dset_ytrain = f.get('train/y_train')
    dset_xtest = f.get('test/x_test')
    dset_ytest = f.get('test/y_test')
    for name in f:
        print(name)
    print('Dsets obtained, now saving as numpy arrays...', flush=True)
    x_trainingdata = dset_xtrain[:]
    y_trainingdata = dset_ytrain[:]
    x_test = dset_xtest[:]
    y_test = dset_ytest[:]
    print('Everything saved, with loop end, file should auto close now', flush=True)

print(x_trainingdata.shape, flush=True)
print(y_trainingdata.shape, flush=True)
print(x_test.shape, flush=True)
print(y_test.shape, flush=True)

x_train, x_val, y_train, y_val = train_test_split(x_trainingdata, y_trainingdata, 
                                                  test_size=0.1, random_state=2)
print('Training data loaded', flush=True)

# Vars for naming results
datasize = x_trainingdata.shape[0] + x_test.shape[0] # Total training data size (train + val + test)
ns = 1024 # N_side for the input data

print('Data size (used for naming results) is: {}'.format(datasize), flush=True)

#-----------------------------
# Model Definition & Compiling
#-----------------------------

print('defining model(hp)...', flush=True) 

# Defining model for optimization of hyperparameters
def build_model(hp):

    # Choice of backbone for encoder/decoder 'arms' of U-net
    model = sm.Unet(hp.Choice('BACKBONE', ['efficientnetb7', 'efficientnetb5', 
                                           'efficientnetb3', 'efficientnetb0', 
                                           'inceptionresnetv2', 'inceptionv3', 
                                           'senet154', 'densenet201', 
                                           'mobilenetv2']), 
                                           input_shape=(256, 256, 2), 
                                           encoder_weights=None, 
                                           classes=2, 
                                           activation='linear')
    
    # Searching over a wide range for the optimal initial learning rate
    lr_test = hp.Float("lr", min_value=1e-5, max_value=5e-2, sampling='log')
    opt = Adam(learning_rate=lr_test)

    # Choice between MSE or MAE as regression losses to optimize
    model.compile(optimizer=opt, loss=hp.Choice('loss', ['MAE', 'MSE']), metrics=['MAE', 'MSE'])

    return model

build_model(kt.HyperParameters())
print('Model built', flush=True)

#-----------------------
# Model Tuner Definition
#-----------------------

es_patience = 5
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                          min_delta=0, 
                                          patience=es_patience, 
                                          restore_best_weights=True)
callbacks = [earlystop]

tuner = kt.Hyperband(
    hypermodel=build_model,
    objective='val_MAE',
    max_epochs=25,
    executions_per_trial=2,
    directory=res_path,
    distribution_strategy=tf.distribute.MirroredStrategy(), # Parallelization strategy
    project_name=pathlib.Path(os.path.join(res_path, 'hptuner/')).mkdir(parents=False, exist_ok=True),
    overwrite=False
)

tuner.search_space_summary()


#-------------
# Model Tuning
#-------------

n_ep=25 # No. of epochs for each combination
b_size=32 # Batch size
run=8 # run number for logging purposes 

tuner.search(
    x=x_train,
    y=y_train,
    validation_data=(x_val, y_val),
    batch_size=b_size,
    epochs=n_ep,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)


#---------------
# Saving Results
#---------------

tuner.results_summary()

models = tuner.get_best_models(num_models=3)
best_model = models[0]
best_model.build(input_shape=(256,256,2))
# best_model.summary()

best_model.save(res_path/'hptuner/best_model.h5')

best2_model = models[1]
best2_model.build(input_shape=(256,256,2))
# best2_model.summary()

best2_model.save(res_path/'hptuner/2ndbest_model.h5')

best3_model = models[2]
best3_model.build(input_shape=(256,256,2))
# best3_model.summary()

best3_model.save(res_path/'hptuner/3rdbest_model.h5')

tuner.results_summary()

print('Execution completed successfully!', flush=True)
